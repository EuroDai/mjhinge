#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

namespace mjhinge {
namespace {

constexpr char kPluginName[] = "mjhinge.hinge_friction";
constexpr const char* kAttributes[] = {"joint", "radial_gain", "axial_gain"};
constexpr int kStateSize = 2;  // radial and axial load from the last solved step

struct Instance {
  int joint;
  int dof;
  int body;
  int coordinator;
  mjtNum radial_gain;
  mjtNum axial_gain;
  mjtNum axis[3] = {};
  bool have_snapshot = false;
  // Only the first instance needs work buffers; all buffers belong to mjData.
  std::vector<mjtNum> velocity;
  std::vector<mjtNum> frictionloss;
  std::vector<mjtNum> tendon_force;
  std::vector<mjtNum> subtree_force;
};

Instance* GetInstance(const mjData* data, int instance) {
  return reinterpret_cast<Instance*>(data->plugin_data[instance]);
}

bool ReadGain(const mjModel* model, int instance, const char* key, mjtNum* value) {
  const char* text = mj_getPluginConfig(model, instance, key);
  *value = 0;
  if (!text || !text[0]) {
    return true;
  }
  errno = 0;
  char* end = nullptr;
  double parsed = std::strtod(text, &end);
  if (end == text || errno == ERANGE) {
    mju_warning("mjhinge.hinge_friction: '%s' must be finite and nonnegative", key);
    return false;
  }
  while (*end && std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  *value = static_cast<mjtNum>(parsed);
  if (*end || !std::isfinite(*value) || *value < 0) {
    mju_warning("mjhinge.hinge_friction: '%s' must be finite and nonnegative", key);
    return false;
  }
  return true;
}

int StateSize(const mjModel*, int) {
  return kStateSize;
}

int Init(const mjModel* model, mjData* data, int instance) {
  const char* joint_name = mj_getPluginConfig(model, instance, "joint");
  if (!joint_name || !joint_name[0]) {
    mju_warning("mjhinge.hinge_friction: config 'joint' is required");
    return -1;
  }

  const int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name);
  if (joint_id < 0) {
    mju_warning("mjhinge.hinge_friction: joint '%s' does not exist", joint_name);
    return -1;
  }
  if (model->jnt_type[joint_id] != mjJNT_HINGE) {
    mju_warning("mjhinge.hinge_friction: joint '%s' must be a hinge", joint_name);
    return -1;
  }
  const int body = model->jnt_bodyid[joint_id];
  if (model->body_jntnum[body] != 1) {
    mju_warning("mjhinge.hinge_friction: joint '%s' must be the body's only joint",
                joint_name);
    return -1;
  }

  mjtNum radial_gain, axial_gain;
  if (!ReadGain(model, instance, "radial_gain", &radial_gain) ||
      !ReadGain(model, instance, "axial_gain", &axial_gain)) {
    return -1;
  }

  int coordinator = instance;
  for (int i = 0; i < model->nplugin; ++i) {
    if (i == instance || model->plugin[i] != model->plugin[instance]) {
      continue;
    }
    if (i < coordinator) {
      coordinator = i;
    }
    const char* other = mj_getPluginConfig(model, i, "joint");
    if (other && mj_name2id(model, mjOBJ_JOINT, other) == joint_id) {
      mju_warning("mjhinge.hinge_friction: duplicate instance for joint '%s'", joint_name);
      return -1;
    }
  }

  try {
    auto info = std::make_unique<Instance>();
    info->joint = joint_id;
    info->dof = model->jnt_dofadr[joint_id];
    info->body = body;
    info->coordinator = coordinator;
    info->radial_gain = radial_gain;
    info->axial_gain = axial_gain;
    if (instance == coordinator) {
      // MuJoCo 3.9 RNE omits spatial tendon forces. Support explicit site
      // routing (including pulley branches); reject unimplemented load sources.
      for (int t = 0; t < model->ntendon; ++t) {
        int adr = model->tendon_adr[t];
        if (model->wrap_type[adr] == mjWRAP_JOINT) continue;
        bool unsupported = model->tendon_stiffness[t] || model->tendon_damping[t] ||
            model->tendon_armature[t] || model->tendon_frictionloss[t] ||
            model->tendon_limited[t];
        for (int k = 0; k < mjNPOLY; ++k) {
          unsupported |= model->tendon_stiffnesspoly[t*mjNPOLY+k] != 0 ||
                         model->tendon_dampingpoly[t*mjNPOLY+k] != 0;
        }
        for (int k = adr; k < adr + model->tendon_num[t]; ++k) {
          unsupported |= model->wrap_type[k] != mjWRAP_SITE &&
                         model->wrap_type[k] != mjWRAP_PULLEY;
        }
        for (int e = 0; e < model->neq; ++e) {
          unsupported |= model->eq_type[e] == mjEQ_TENDON &&
              (model->eq_obj1id[e] == t || model->eq_obj2id[e] == t);
        }
        for (int a = 0; a < model->nu; ++a) {
          if (model->actuator_trntype[a] != mjTRN_TENDON ||
              model->actuator_trnid[2*a] != t) continue;
          unsupported |= model->actuator_damping[a] != 0 ||
                         model->actuator_armature[a] != 0;
          for (int k = 0; k < mjNPOLY; ++k) {
            unsupported |= model->actuator_dampingpoly[a*mjNPOLY+k] != 0;
          }
        }
        if (unsupported) {
          mju_warning("mjhinge: spatial tendons currently require site/pulley routing "
                      "and actuator forces only (no passive/limit/equality loads)");
          return -1;
        }
      }
      info->velocity.resize(model->nv);
      info->frictionloss.resize(model->nv);
      info->tendon_force.resize(model->ntendon);
      info->subtree_force.resize(3 * model->nbody);
    }
    data->plugin_data[instance] = reinterpret_cast<uintptr_t>(info.release());
  } catch (const std::bad_alloc&) {
    mju_warning("mjhinge.hinge_friction: could not allocate instance data");
    return -1;
  }
  return 0;
}

void Destroy(mjData* data, int instance) {
  delete GetInstance(data, instance);
  data->plugin_data[instance] = 0;
}

void Copy(mjData* dest, const mjModel*, const mjData* src, int instance) {
  *GetInstance(dest, instance) = *GetInstance(src, instance);
}

void Reset(const mjModel*, mjtNum* state, void* plugin_data, int) {
  for (int i = 0; i < kStateSize; ++i) {
    state[i] = 0;
  }
  if (plugin_data) {
    static_cast<Instance*>(plugin_data)->have_snapshot = false;
  }
}

mjtNum Limit(const mjModel* model, const mjData* data, int instance,
             const Instance& info) {
  const mjtNum* load = data->plugin_state + model->plugin_stateadr[instance];
  const mjtNum limit = model->dof_frictionloss[info.dof] +
                      info.radial_gain * load[0] + info.axial_gain * load[1];
  if (!std::isfinite(limit) || limit < 0) {
    mju_error("mjhinge.hinge_friction: invalid friction limit");
  }
  return limit;
}

int FrictionRow(const mjData* data, int dof) {
  for (int row = data->ne; row < data->ne + data->nf; ++row) {
    if (data->efc_type[row] == mjCNSTR_FRICTION_DOF && data->efc_id[row] == dof) {
      return row;
    }
  }
  return -1;
}

bool Enabled(const mjModel* model) {
  const int flags = model->opt.disableflags;
  return !(flags & (mjDSBL_CONSTRAINT | mjDSBL_FRICTIONLOSS)) &&
         !((flags & mjDSBL_SPRING) && (flags & mjDSBL_DAMPER));
}

void Compute(const mjModel* model, mjData* data, int instance, int capability) {
  Instance* workspace = GetInstance(data, instance);
  if (capability != mjPLUGIN_PASSIVE || instance != workspace->coordinator ||
      !Enabled(model)) {
    return;
  }

  bool rebuild = false;
  bool active = false;
  for (int i = 0; i < model->nplugin; ++i) {
    if (model->plugin[i] != model->plugin[instance]) {
      continue;
    }
    Instance* info = GetInstance(data, i);
    if (info->radial_gain == 0 && info->axial_gain == 0) {
      continue;
    }
    active = true;
    for (int k = 0; k < 3; ++k) {
      info->axis[k] = data->xaxis[3 * info->joint + k];
    }
    // Positive native frictionloss already created this row. A zero native
    // value needs a new row once the lagged load becomes positive.
    if (Limit(model, data, i, *info) > 0 && FrictionRow(data, info->dof) < 0 &&
        (!(model->opt.enableflags & mjENBL_SLEEP) ||
         data->tree_awake[model->dof_treeid[info->dof]])) {
      rebuild = true;
    }
  }
  if (!active) {
    return;
  }
  for (int k = 0; k < model->nv; ++k) {
    workspace->velocity[k] = data->qvel[k];
  }
  workspace->have_snapshot = true;

  if (rebuild) {
    for (int k = 0; k < model->nv; ++k) {
      workspace->frictionloss[k] = model->dof_frictionloss[k];
    }
    for (int i = 0; i < model->nplugin; ++i) {
      if (model->plugin[i] == model->plugin[instance]) {
        const Instance& info = *GetInstance(data, i);
        workspace->frictionloss[info.dof] = Limit(model, data, i, info);
      }
    }
    // A local view changes only the inputs to constraint construction. The
    // shared mjModel and its native frictionloss values remain untouched.
    mjModel view = *model;
    view.dof_frictionloss = workspace->frictionloss.data();
    mj_makeConstraint(&view, data);
    mj_island(&view, data);
    mj_projectConstraint(&view, data);
  }

  // This callback runs after constraint construction and before constraint
  // references/solving. Let MuJoCo choose the static/sliding friction torque.
  for (int i = 0; i < model->nplugin; ++i) {
    if (model->plugin[i] != model->plugin[instance]) {
      continue;
    }
    const Instance& info = *GetInstance(data, i);
    if (info.radial_gain == 0 && info.axial_gain == 0) {
      continue;
    }
    const int row = FrictionRow(data, info.dof);
    if (row >= 0) {
      const mjtNum limit = Limit(model, data, i, info);
      data->efc_frictionloss[row] = limit;
      // Primal solvers use the island-sorted copy made in mj_fwdPosition.
      if (data->nisland > 0 && data->map_efc2iefc) {
        data->iefc_frictionloss[data->map_efc2iefc[row]] = limit;
      }
    }
  }
}

// Recover physical site forces omitted by mj_rnePostConstraint in MuJoCo 3.9.
// These are used only for measuring bearing loads, never applied a second time.
void TendonLoads(const mjModel* model, const mjData* data, Instance* workspace) {
  auto& tension = workspace->tendon_force;
  auto& force = workspace->subtree_force;
  std::fill(tension.begin(), tension.end(), 0);
  std::fill(force.begin(), force.end(), 0);
  if (!(model->opt.disableflags & mjDSBL_ACTUATION)) {
    for (int a = 0; a < model->nu; ++a) {
      if (model->actuator_trntype[a] == mjTRN_TENDON) {
        tension[model->actuator_trnid[2*a]] +=
            data->actuator_force[a] * model->actuator_gear[6*a];
      }
    }
  }
  for (int t = 0; t < model->ntendon; ++t) {
    int adr = model->tendon_adr[t];
    if (model->wrap_type[adr] == mjWRAP_JOINT) continue;
    mjtNum scalar = tension[t];
    // actuator_force already includes native actuator/tendon force clamping.
    mjtNum divisor = 1;
    int previous = -1;
    for (int w = adr; w < adr + model->tendon_num[t]; ++w) {
      if (model->wrap_type[w] == mjWRAP_PULLEY) {
        divisor = model->wrap_prm[w];
        previous = -1;
        continue;
      }
      int site = model->wrap_objid[w];
      if (previous >= 0) {
        mjtNum delta[3], length = 0;
        for (int k = 0; k < 3; ++k) {
          delta[k] = data->site_xpos[3*site+k] - data->site_xpos[3*previous+k];
          length = std::hypot(length, delta[k]);
        }
        if (length > mjMINVAL) {
          int b0 = model->site_bodyid[previous], b1 = model->site_bodyid[site];
          for (int k = 0; k < 3; ++k) {
            mjtNum f = -scalar * delta[k] / (divisor * length);
            force[3*b0+k] += f;
            force[3*b1+k] -= f;
          }
        }
      }
      previous = site;
    }
  }
  for (mjtSize b = model->nbody-1; b > 0; --b) {
    for (int k = 0; k < 3; ++k) {
      force[3*model->body_parentid[b]+k] += force[3*b+k];
    }
  }
}

void Advance(const mjModel* model, mjData* data, int instance) {
  Instance* workspace = GetInstance(data, instance);
  if (instance != workspace->coordinator || !Enabled(model) ||
      !workspace->have_snapshot) {
    return;
  }
  workspace->have_snapshot = false;

  // advance() runs after integration. Kinematics, qacc and constraint forces
  // still describe the last solve, but qvel has advanced. Use that solve's
  // saved velocity when reconstructing its reaction, including for RK4.
  mjData view = *data;
  view.qvel = workspace->velocity.data();
  mj_rnePostConstraint(model, &view);
  data->flg_rnepost = 1;
  TendonLoads(model, data, workspace);

  for (int i = 0; i < model->nplugin; ++i) {
    if (model->plugin[i] != model->plugin[instance]) {
      continue;
    }
    const Instance& info = *GetInstance(data, i);
    if (info.radial_gain == 0 && info.axial_gain == 0) {
      continue;
    }
    mjtNum force[3];
    for (int k = 0; k < 3; ++k) {
      force[k] = data->cfrc_int[6*info.body+3+k] - workspace->subtree_force[3*info.body+k];
    }
    mjtNum axial = 0;
    for (int k = 0; k < 3; ++k) {
      axial += force[k] * info.axis[k];
    }
    mjtNum radial = 0;
    for (int k = 0; k < 3; ++k) {
      radial = std::hypot(radial, force[k] - axial * info.axis[k]);
    }
    if (!std::isfinite(radial) || !std::isfinite(axial)) {
      mju_error("mjhinge.hinge_friction: non-finite joint reaction");
    }
    mjtNum* state = data->plugin_state + model->plugin_stateadr[i];
    state[0] = radial;
    state[1] = std::abs(axial);
  }
}

}  // namespace

void RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = kPluginName;
  plugin.nattribute = sizeof(kAttributes) / sizeof(kAttributes[0]);
  plugin.attributes = kAttributes;
  plugin.capabilityflags = mjPLUGIN_PASSIVE;
  plugin.nstate = StateSize;
  plugin.init = Init;
  plugin.destroy = Destroy;
  plugin.copy = Copy;
  plugin.reset = Reset;
  plugin.compute = Compute;
  plugin.advance = Advance;
  mjp_registerPlugin(&plugin);
}

}  // namespace mjhinge

mjPLUGIN_LIB_INIT(mjhinge) {
  mjhinge::RegisterPlugin();
}

