import jax.numpy as jnp
# Stores some common damage dictionaries for the hexapod

def get_damage_dict(in_var):
    # If user specifies damage with a dictionary
    if type(in_var) == dict:
        arr = [200.0] * 18

        for index in in_var.keys():
            arr[index] = in_var[index]
    
    # If user specifies damage with an array
    else:
        arr = in_var

    damage_dict = {
            "body_leg_0": float(arr[0]),
            "leg_0_1_2": float(arr[1]),
            "leg_0_2_3": float(arr[2]),

            "body_leg_1": float(arr[3]),
            "leg_1_1_2": float(arr[4]),
            "leg_1_2_3": float(arr[5]),

            "body_leg_2": float(arr[6]),
            "leg_2_1_2": float(arr[7]),
            "leg_2_2_3": float(arr[8]),

            "body_leg_3": float(arr[9]),
            "leg_3_1_2": float(arr[10]),
            "leg_3_2_3": float(arr[11]),

            "body_leg_4": float(arr[12]),
            "leg_4_1_2": float(arr[13]),
            "leg_4_2_3": float(arr[14]),

            "body_leg_5": float(arr[15]),
            "leg_5_1_2": float(arr[16]),
            "leg_5_2_3": float(arr[17]),
        }

    return damage_dict

# Some common damage_dicts
all_actuators_broken = get_damage_dict([0] * 18)

if __name__ == "__main__":
    import jax.numpy as jnp
    print(get_damage_dict({1: 0, 5:0}))