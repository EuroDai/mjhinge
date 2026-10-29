"""Device-only friction limits, lagged bearing loads and independent resets."""

import warp as wp


@wp.struct
class Parameters:
    joint: int
    dof: int
    body: int
    qpos: int
    active: int
    base: float
    radial_gain: float
    axial_gain: float
    static_extra: float
    stribeck_velocity: float
    radial_peak_gain: float
    radial_peak_angle: float
    radial_peak_width: float


@wp.kernel
def limits(parameters: wp.array[Parameters], loads: wp.array2d[wp.vec2],
           qpos: wp.array2d[float], qvel: wp.array2d[float],
           enabled: int, values: wp.array2d[float]):
    world, hinge = wp.tid()
    p = parameters[hinge]
    value = p.base
    if enabled != 0 and p.active != 0:
        coordinate = (qpos[world, p.qpos] - p.radial_peak_angle) / p.radial_peak_width
        radial = p.radial_gain + p.radial_peak_gain * wp.exp(-0.5 * coordinate * coordinate)
        speed = qvel[world, p.dof] / p.stribeck_velocity
        load = loads[world, hinge]
        value += radial * load[0] + p.axial_gain * load[1] + p.static_extra * wp.exp(-speed * speed)
    values[world, hinge] = value


@wp.kernel
def update_rows(parameters: wp.array[Parameters], values: wp.array2d[float],
                nefc: wp.array[int], types: wp.array2d[int], ids: wp.array2d[int],
                friction: wp.array2d[float]):
    world, row = wp.tid()
    if row < nefc[world] and types[world, row] == 1:  # mjCNSTR_FRICTION_DOF
        for hinge in range(parameters.shape[0]):
            p = parameters[hinge]
            if p.active != 0 and ids[world, row] == p.dof:
                friction[world, row] = values[world, hinge]


@wp.kernel
def advance(parameters: wp.array[Parameters], force: wp.array2d[wp.spatial_vector],
            axis: wp.array2d[wp.vec3], loads: wp.array2d[wp.vec2]):
    world, hinge = wp.tid()
    p = parameters[hinge]
    if p.active != 0:
        reaction = force[world, p.body]
        f = wp.vec3(reaction[3], reaction[4], reaction[5])
        a = axis[world, p.joint]
        axial = wp.dot(f, a)
        loads[world, hinge] = wp.vec2(wp.length(f - axial * a), wp.abs(axial))


@wp.kernel
def reset_history(mask: wp.array[bool], loads: wp.array2d[wp.vec2]):
    world, hinge = wp.tid()
    if mask[world]:
        loads[world, hinge] = wp.vec2(0.0)
