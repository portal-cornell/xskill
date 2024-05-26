import numpy as np


def ease_in_out_sine(x):
    return -(np.cos(np.pi * x) - 1) / 2


def ease_linear(x):
    return x


def ease_out_quad(x):
    return 1 - (1 - x) * (1 - x)


def create_action(action, duration=[20], pause=[10], completion=1, ease=ease_out_quad):
    """
    Creates a dictionary to represent the action

    Parameters:
    ----------
    action (string): the name of the action
    duration (int list) : the number of environment steps for each subaction. Subaction 1 takes duration[0] time.
    pause (int lis): the number of environment steps to pause after completing each subaction
    completion (float): value from 0 to 1 inclusive representing the fraction of opening for the microwave, slide cabinet, and hinge cabinet action. Is 1 for all other tasks
    ease: function determining interpolation of action
    """
    return {
        "action": action,
        "durations": duration,
        "pause": pause,
        "completion": completion,
        "ease": ease,
    }


# v1
lift_kettle_action = create_action(
    "lift kettle", duration = [10, 10, 10], pause = [30, 0, 0], ease=ease_linear
)
top_burner_action = create_action(
    "top burner", duration = [30], pause = [30], ease=ease_linear
)
bottom_burner_action = create_action(
    "bottom burner", duration = [30], pause = [30], ease=ease_linear
)
slide_action = create_action(
    "slide cabinet", duration = [30], pause = [30], ease=ease_linear
)
hinge_action = create_action(
    "hinge cabinet", duration = [30], pause = [30], ease=ease_linear
)
microwave_action = create_action(
    "microwave", duration = [30], pause = [30], ease=ease_linear
)
light_action = create_action(
    "light switch", duration = [30], pause = [30], ease=ease_linear
)

# v0 - invisible
# lift_kettle_action = create_action(
#     "lift kettle", duration = [20, 10, 20], pause = [30, 0, 0], ease=ease_linear
# )
# top_burner_action = create_action(
#     "top burner", duration = [30], pause = [30], ease=ease_linear
# )
# bottom_burner_action = create_action(
#     "bottom burner", duration = [30], pause = [30], ease=ease_linear
# )
# slide_action = create_action(
#     "slide cabinet", duration = [20], pause = [25], ease=ease_linear
# )
# hinge_action = create_action(
#     "hinge cabinet", duration = [30], pause = [25], ease=ease_linear
# )
# microwave_action = create_action(
#     "microwave", duration = [30], pause = [35], ease=ease_linear
# )
# light_action = create_action(
#     "light switch", duration = [10], pause = [35], ease=ease_linear
# )


# top_burner_action = create_action("top burner")
# bottom_burner_action = create_action("bottom burner")
# microwave_action = create_action("microwave")
kettle_action = create_action("kettle")
# light_action = create_action("light switch")
# slide_action = create_action("slide cabinet")
# hinge_action = create_action("hinge cabinet")

# v1
fast_top_burner_action = create_action("top burner", duration=[10])
fast_bottom_burner_action = create_action("bottom burner", duration=[10])
fast_microwave_action = create_action("microwave", duration=[10])
fast_kettle_action = create_action("kettle", duration=[10])
fast_light_action = create_action("light switch", duration=[10])
fast_slide_action = create_action("slide cabinet", duration=[10])
fast_hinge_action = create_action("hinge cabinet", duration=[10])

full_hinge_action = create_action("full hinge cabinet")
full_slide_action = create_action("full slide cabinet")
full_microwave_action = create_action("full microwave")
half_microwave_action = create_action("full microwave", completion=0.5)
half_hinge_action = create_action("full hinge cabinet", completion=0.5)
quarter_slide_action = create_action("full slide cabinet", completion=0.25)
