import config as c
import numpy as np

def p_count(layout): #[Input, Hidden Layer 1, ....; Output]
    lay_param = 0
    for i in range(len(layout) - 1):
        lay_param += layout[i] * layout[i + 1] + layout[i + 1]
    return lay_param



def P(env):
    P_coeff = 0.7
    return  2 * (env.theta-np.pi) * P_coeff * c.FORCE_MAG / np.pi

def PI(env):
    P_coeff = 0.7
    I_coeff = 0.2

def MLP(env, parameters, layout, activation = None):
    #parameter shape (Pop_size, Parameter count), layout array [Input_size, HL2_size...., HLM_size]
    lay_param = p_count(layout)

    if len(parameters[0,:]) != lay_param:
        return
    else:
        pop_size = parameters.shape[0]
        offset = 0
        weights_and_biases = []

        for i in range(len(layout)-1):
            w_size = layout[i] * layout[i+1]
            b_size = layout[i+1]

            W = parameters[:, offset:offset+w_size].reshape(pop_size, layout[i], layout[i+1])
            offset += w_size

            b = parameters[:, offset:offset+b_size]
            offset += b_size

            weights_and_biases.append((W, b))

    if activation is None:
        activation = np.tanh

        # input features -- adjust this if you want different inputs; must match layout[0]
    state = np.stack(
        [np.sin(env.theta), np.cos(env.theta), env.theta_dot, env.x / env.x_threshold, env.x_dot],
        axis=1
    )

    h = state
    for W, b in weights_and_biases:
        h = activation(np.einsum('ni,nij->nj', h, W) + b)

    force = h[:, 0] * env.force_mag
    return force