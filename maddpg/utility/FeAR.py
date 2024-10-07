# Libraries
import numpy as np
import copy
from typing import Callable
from typing import Callable, List, Sequence, Tuple, Union

# General Settings
# DEFAULT_TARGET_SPEEDS = [20, 25, 30] # m/s
# DiscreteMetaAction = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
# dt = 1 / 15
Vector = Union[np.ndarray, Sequence[float]]



def project_polygon(polygon: Vector, axis: Vector) -> tuple[float, float]:
    min_p, max_p = None, None
    for p in polygon:
        projected = p.dot(axis)
        if min_p is None or projected < min_p:
            min_p = projected
        if max_p is None or projected > max_p:
            max_p = projected
    return min_p, max_p


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
    """
    Calculate the distance between [minA, maxA] and [minB, maxB]
    The distance will be negative if the intervals overlap
    """
    return min_b - max_a if min_a < min_b else min_a - max_b


def are_polygons_intersecting(
    a: Vector, b: Vector, displacement_a: Vector, displacement_b: Vector
):
    """
    Checks if the two polygons are intersecting.

    See https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

    :param a: polygon A, as a list of [x, y] points
    :param b: polygon B, as a list of [x, y] points
    :param displacement_a: velocity of the polygon A
    :param displacement_b: velocity of the polygon B
    :return: are intersecting, will intersect, translation vector
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:]):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # center difference
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


def _is_colliding(ego, other, delta_speed, dt=1/15):
    # Fast spherical pre-check
    if (
        np.linalg.norm(other.position - ego.position)
        > (ego.diagonal + other.diagonal) / 2 + ego.speed * dt
    ):
        return (
            False,
            False,
            np.zeros(
                2,
            ),
        )
    # Accurate rectangular check
    return are_polygons_intersecting(
        ego.polygon(), other.polygon(), (ego.velocity +delta_speed) * dt, other.velocity * dt
    )


def count_FeasibleActions(i, j, action, env, DiscreteMetaAction={0: "SLOWER", 1: "IDLE", 2: "FASTER"}):
    '''
    Function that counts the number of feasible actions of agent "others" given the movement of agent "ego"

    return: the number of feasible actions of agent others given the movement of agent ego, int
    '''

    if type(env.unwrapped.road.vehicles[j]) == IDMVehicle:

        environment = copy.deepcopy(env)
        _, _, _, _, info = environment.step(tuple(action)
        agent_j = environment.unwrapped.road.vehicles[j]

        if agent_j.crashed == True:
            del environment
            return 0
        else:
            del environment
            return 1

    elif type(env.unwrapped.road.vehicles[j]) == MDPVehicle:

        count = 0

        for action_j in range(len(DiscreteMetaAction)):

            action[j] = action_j
            environment = copy.deepcopy(env)
            _, _, _, _, info = environment.step(tuple(action)
            agent_j = environment.unwrapped.road.vehicles[j]

            if agent_j.crashed == False:
                del environment
                count += 1
            else:
                del environment

        return count

    else:
        raise Exception("Unknown type of vehicle.")

    return


def cal_FeAR_ij(i, j, action, MdR, before_action_env, epsilon = 1e-6):
    '''
    Function that calculates the FeAR value of agent i on agent j, i.e. FeAR_ij

    before_action_env: object, environment before taking the actions
    epsilon: float, add on denominator to avoid dividing by 0

    return: FrAR_ij, float 
    '''

    if i == j or action[i] == MdR[i]:
        return 0.0

    else:
        action_MdRi = copy.deepcopy(action)
        action_MdRi[i] = MdR[i]
            
        n_FeasibleActions_MdRi_j = count_FeasibleActions(i, j, action_MdRi, before_action_env)
        n_FeasibleActions_Actioni_j = count_FeasibleActions(i, j, action, before_action_env)

        FeAR_ij = np.clip( ( (n_FeasibleActions_MdRi_j - n_FeasibleActions_Actioni_j) / (n_FeasibleActions_MdRi_j + epsilon) ), -1, 1)

        return FeAR_ij


def cal_MdR(agents, DEFAULT_TARGET_SPEEDS=[0, 4.5, 9]):
    '''
    Function that calculate the MdR of all agents in the environment

    DEFAULT_TARGET_SPEEDS = [0, 4.5, 9] m/s
    DiscreteMetaAction = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}

    return: list of intergers, len = n_agents
    '''

    MdR = []
    for agent in agents:
        MdR_i = 2 if agent.speed <= (DEFAULT_TARGET_SPEEDS[0] + 1.0) else (0 if agent.speed >= (DEFAULT_TARGET_SPEEDS[2] + 1.0) else 1)
        MdR.append(MdR_i)

    return MdR

