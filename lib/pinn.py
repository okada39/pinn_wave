import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the wave equation.

    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        c: wave velocity.
        grads: gradient layer.
    """

    def __init__(self, network, c=1):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            c: wave velocity. Default is 1.
        """

        self.network = network
        self.c = c
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the wave equation.

        Returns:
            PINN model for the projectile motion with
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition,
                         (t, x=bounds) relative to boundary condition ],
                output: [ u(t,x) relative to equation,
                          u(t=0, x) relative to initial condition,
                          du_dt(t=0, x) relative to initial derivative of t,
                          u(t, x=bounds) relative to boundary condition ]
        """

        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input: (t=0, x)
        tx_ini = tf.keras.layers.Input(shape=(2,))
        # boundary condition input: (t, x=-1) or (t, x=+1)
        tx_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        _, _, _, d2u_dt2, d2u_dx2 = self.grads(tx_eqn)

        # equation output being zero
        u_eqn = d2u_dt2 - self.c*self.c * d2u_dx2
        # initial condition output
        u_ini, du_dt_ini, _, _, _ = self.grads(tx_ini)
        # boundary condition output
        u_bnd = self.network(tx_bnd)  # dirichlet
        #_, _, u_bnd, _, _ = self.grads(tx_bnd)  # neumann

        # build the PINN model for the wave equation
        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_ini, tx_bnd],
            outputs=[u_eqn, u_ini, du_dt_ini, u_bnd])
