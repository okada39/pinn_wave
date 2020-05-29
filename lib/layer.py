import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for the wave equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, tx):
        """
        Computing 1st and 2nd derivatives for the wave equation.

        Args:
            tx: input variables (t, x).

        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
        """

        with tf.GradientTape() as g:
            g.watch(tx)
            with tf.GradientTape() as gg:
                gg.watch(tx)
                u = self.model(tx)
            du_dtx = gg.batch_jacobian(u, tx)
            du_dt = du_dtx[..., 0]
            du_dx = du_dtx[..., 1]
        d2u_dtx2 = g.batch_jacobian(du_dtx, tx)
        d2u_dt2 = d2u_dtx2[..., 0, 0]
        d2u_dx2 = d2u_dtx2[..., 1, 1]

        return u, du_dt, du_dx, d2u_dt2, d2u_dx2
