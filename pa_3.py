import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. MINI-BATCH GRADIENT DESCENT
# ============================================================

class MyMiniBatchGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, w, gradient):
        w = w - self.learning_rate * gradient
        return w


# Convex function
def convex_function(w):
    w1, w2 = w
    a1, a2 = 2.0, 5.0
    b1, b2 = 3.0, -2.0
    c = 10.0

    return a1 * (w1 - b1)**2 + a2 * (w2 - b2)**2 + c


# Gradient of convex function
def convex_gradient(w):
    w1, w2 = w
    a1, a2 = 2.0, 5.0
    b1, b2 = 3.0, -2.0

    dw1 = 2 * a1 * (w1 - b1)
    dw2 = 2 * a2 * (w2 - b2)

    return np.array([dw1, dw2])


# Optimization
optimizer = MyMiniBatchGD(learning_rate=0.05)

w = np.array([0.0, 0.0])

history = [w.copy()]
loss_history = [convex_function(w)]

for i in range(100):

    gradient = convex_gradient(w)

    w = optimizer.update(w, gradient)

    history.append(w.copy())
    loss_history.append(convex_function(w))


history = np.array(history)


print("Mini-Batch Gradient Descent")
print("Final w =", w)
print("Final function value =", convex_function(w))


# Cost plot
plt.figure(figsize=(7, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Mini-Batch Gradient Descent - Convergence")
plt.grid(True)
plt.show()


# Contour plot + trajectory
w1_values = np.linspace(-1, 5, 200)
w2_values = np.linspace(-4, 2, 200)

W1, W2 = np.meshgrid(w1_values, w2_values)

Z = (
    2 * (W1 - 3)**2
    + 5 * (W2 + 2)**2
    + 10
)

plt.figure(figsize=(7, 6))

plt.contour(W1, W2, Z, levels=20)

plt.plot(
    history[:, 0],
    history[:, 1],
    marker='o',
    markersize=3,
    label="Optimization Path"
)

plt.scatter(3, -2, marker='*', s=150, label="Minimum")

plt.xlabel("w1")
plt.ylabel("w2")
plt.title("Mini-Batch Gradient Descent on Convex Function")
plt.legend()
plt.grid(True)
plt.show()
--===========================================================================================================


# ============================================================
# 2. SGD WITH MOMENTUM
# ============================================================

class MyMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros(2)

    def update(self, w, gradient):

        self.velocity = (
            self.momentum * self.velocity
            + gradient
        )

        w = w - self.learning_rate * self.velocity

        return w


optimizer = MyMomentum(
    learning_rate=0.05,
    momentum=0.9
)

w = np.array([0.0, 0.0])

history = [w.copy()]
loss_history = [convex_function(w)]

for i in range(100):

    gradient = convex_gradient(w)

    w = optimizer.update(w, gradient)

    history.append(w.copy())
    loss_history.append(convex_function(w))


history = np.array(history)


print("SGD with Momentum")
print("Final w =", w)
print("Final function value =", convex_function(w))


# Cost plot
plt.figure(figsize=(7, 5))

plt.plot(loss_history)

plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("SGD with Momentum - Convergence")

plt.grid(True)
plt.show()


# Contour plot
plt.figure(figsize=(7, 6))

plt.contour(W1, W2, Z, levels=20)

plt.plot(
    history[:, 0],
    history[:, 1],
    marker='o',
    markersize=3,
    label="Optimization Path"
)

plt.scatter(
    3, -2,
    marker='*',
    s=150,
    label="Minimum"
)

plt.xlabel("w1")
plt.ylabel("w2")

plt.title("SGD with Momentum on Convex Function")

plt.legend()
plt.grid(True)

plt.show()
===================================================================
# ============================================================
# 3. NESTEROV ACCELERATED GRADIENT
# ============================================================

class MyNesterov:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros(2)

    def update(self, w, gradient_function):

        # Look-ahead position
        lookahead = (
            w
            + self.momentum * self.velocity
        )

        # Gradient at look-ahead position
        gradient = gradient_function(lookahead)

        # Update velocity
        self.velocity = (
            self.momentum * self.velocity
            - self.learning_rate * gradient
        )

        # Update parameters
        w = w + self.velocity

        return w


optimizer = MyNesterov(
    learning_rate=0.05,
    momentum=0.9
)

w = np.array([0.0, 0.0])

history = [w.copy()]
loss_history = [convex_function(w)]

for i in range(100):

    w = optimizer.update(
        w,
        convex_gradient
    )

    history.append(w.copy())
    loss_history.append(convex_function(w))


history = np.array(history)


print("Nesterov Accelerated Gradient")
print("Final w =", w)
print("Final function value =", convex_function(w))


# Cost plot
plt.figure(figsize=(7, 5))

plt.plot(loss_history)

plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Nesterov Accelerated Gradient - Convergence")

plt.grid(True)
plt.show()


# Contour plot
plt.figure(figsize=(7, 6))

plt.contour(W1, W2, Z, levels=20)

plt.plot(
    history[:, 0],
    history[:, 1],
    marker='o',
    markersize=3,
    label="Optimization Path"
)

plt.scatter(
    3, -2,
    marker='*',
    s=150,
    label="Minimum"
)

plt.xlabel("w1")
plt.ylabel("w2")

plt.title("Nesterov Accelerated Gradient on Convex Function")

plt.legend()
plt.grid(True)

plt.show()
=========================================================================
# ============================================================
# 4. ADAM OPTIMIZER
# ============================================================

class MyAdam:
    def __init__(
        self,
        learning_rate=0.05,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def update(self, w, gradient):

        self.t += 1

        # First moment
        self.m = (
            self.beta1 * self.m
            + (1 - self.beta1) * gradient
        )

        # Second moment
        self.v = (
            self.beta2 * self.v
            + (1 - self.beta2) * gradient**2
        )

        # Bias correction
        m_hat = (
            self.m
            / (1 - self.beta1**self.t)
        )

        v_hat = (
            self.v
            / (1 - self.beta2**self.t)
        )

        # Parameter update
        w = w - (
            self.learning_rate
            * m_hat
            / (np.sqrt(v_hat) + self.epsilon)
        )

        return w


optimizer = MyAdam(
    learning_rate=0.05
)

w = np.array([0.0, 0.0])

history = [w.copy()]
loss_history = [convex_function(w)]

for i in range(100):

    gradient = convex_gradient(w)

    w = optimizer.update(
        w,
        gradient
    )

    history.append(w.copy())
    loss_history.append(convex_function(w))


history = np.array(history)


print("Adam Optimizer")
print("Final w =", w)
print("Final function value =", convex_function(w))


# Cost plot
plt.figure(figsize=(7, 5))

plt.plot(loss_history)

plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Adam Optimizer - Convergence")

plt.grid(True)
plt.show()


# Contour plot
plt.figure(figsize=(7, 6))

plt.contour(W1, W2, Z, levels=20)

plt.plot(
    history[:, 0],
    history[:, 1],
    marker='o',
    markersize=3,
    label="Optimization Path"
)

plt.scatter(
    3, -2,
    marker='*',
    s=150,
    label="Minimum"
)

plt.xlabel("w1")
plt.ylabel("w2")

plt.title("Adam Optimizer on Convex Function")

plt.legend()
plt.grid(True)

plt.show()
===============================================================
# ============================================================
# ROSENBROCK FUNCTION
# Apply all 4 optimizers
# ============================================================

def rosenbrock(w):
    w1, w2 = w

    return (
        100 * (w2 - w1**2)**2
        + (1 - w1)**2
    )


def rosenbrock_gradient(w):
    w1, w2 = w

    dw1 = (
        -400 * w1 * (w2 - w1**2)
        - 2 * (1 - w1)
    )

    dw2 = (
        200 * (w2 - w1**2)
    )

    return np.array([dw1, dw2])


# ------------------------------------------------------------
# Starting point
# ------------------------------------------------------------

initial_w = np.array([-1.5, 1.5])


# ------------------------------------------------------------
# Function to run optimizer
# ------------------------------------------------------------

def run_rosenbrock(optimizer, name, iterations=5000):

    w = initial_w.copy()

    history = [w.copy()]
    loss_history = [rosenbrock(w)]

    for i in range(iterations):

        if isinstance(optimizer, MyNesterov):

            w = optimizer.update(
                w,
                rosenbrock_gradient
            )

        else:

            gradient = rosenbrock_gradient(w)

            w = optimizer.update(
                w,
                gradient
            )

        history.append(w.copy())
        loss_history.append(rosenbrock(w))

    history = np.array(history)

    print("\n", name)
    print("Final w =", w)
    print("Final function value =", rosenbrock(w))

    return history, loss_history


# ============================================================
# 1. Mini-Batch Gradient Descent
# ============================================================

gd = MyMiniBatchGD(
    learning_rate=0.0005
)

gd_history, gd_loss = run_rosenbrock(
    gd,
    "Mini-Batch Gradient Descent"
)


# ============================================================
# 2. Momentum
# ============================================================

momentum = MyMomentum(
    learning_rate=0.0005,
    momentum=0.9
)

momentum_history, momentum_loss = run_rosenbrock(
    momentum,
    "SGD with Momentum"
)


# ============================================================
# 3. Nesterov
# ============================================================

nesterov = MyNesterov(
    learning_rate=0.0005,
    momentum=0.9
)

nesterov_history, nesterov_loss = run_rosenbrock(
    nesterov,
    "Nesterov Accelerated Gradient"
)


# ============================================================
# 4. Adam
# ============================================================

adam = MyAdam(
    learning_rate=0.01
)

adam_history, adam_loss = run_rosenbrock(
    adam,
    "Adam"
)
=======================================================================================================


# ============================================================
# ROSENBROCK - 3D SURFACE + TRAJECTORIES
# ============================================================

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot_surface(
    W1,
    W2,
    Z,
    alpha=0.5
)

ax.plot(
    gd_history[:, 0],
    gd_history[:, 1],
    [
        rosenbrock(x)
        for x in gd_history
    ],
    label="Mini-Batch GD"
)

ax.plot(
    momentum_history[:, 0],
    momentum_history[:, 1],
    [
        rosenbrock(x)
        for x in momentum_history
    ],
    label="Momentum"
)

ax.plot(
    nesterov_history[:, 0],
    nesterov_history[:, 1],
    [
        rosenbrock(x)
        for x in nesterov_history
    ],
    label="Nesterov"
)

ax.plot(
    adam_history[:, 0],
    adam_history[:, 1],
    [
        rosenbrock(x)
        for x in adam_history
    ],
    label="Adam"
)

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("f(w1,w2)")

ax.set_title(
    "Rosenbrock Function - Optimization Trajectories"
)

ax.legend()

plt.show()
==========================================================================
# ============================================================
# BEALE FUNCTION
# ============================================================

def beale(w):

    w1, w2 = w

    term1 = (
        1.5
        - w1
        + w1 * w2
    )

    term2 = (
        2.25
        - w1
        + w1 * w2**2
    )

    term3 = (
        2.625
        - w1
        + w1 * w2**3
    )

    return (
        term1**2
        + term2**2
        + term3**2
    )


def beale_gradient(w):

    w1, w2 = w

    t1 = (
        1.5
        - w1
        + w1 * w2
    )

    t2 = (
        2.25
        - w1
        + w1 * w2**2
    )

    t3 = (
        2.625
        - w1
        + w1 * w2**3
    )

    dw1 = (
        2 * t1 * (-1 + w2)
        + 2 * t2 * (-1 + w2**2)
        + 2 * t3 * (-1 + w2**3)
    )

    dw2 = (
        2 * t1 * w1
        + 2 * t2 * 2 * w1 * w2
        + 2 * t3 * 3 * w1 * w2**2
    )

    return np.array([
        dw1,
        dw2
    ])
  ========================================================================================
# ============================================================
# APPLY ALL 4 OPTIMIZERS TO BEALE
# ============================================================

initial_w = np.array([1.0, 1.0])


def run_beale(optimizer, name, iterations=5000):

    w = initial_w.copy()

    history = [w.copy()]
    loss_history = [beale(w)]

    for i in range(iterations):

        if isinstance(optimizer, MyNesterov):

            w = optimizer.update(
                w,
                beale_gradient
            )

        else:

            gradient = beale_gradient(w)

            w = optimizer.update(
                w,
                gradient
            )

        history.append(w.copy())
        loss_history.append(beale(w))

    history = np.array(history)

    print("\n", name)
    print("Final w =", w)
    print("Final function value =", beale(w))

    return history, loss_history


# ------------------------------------------------------------
# Mini-Batch GD
# ------------------------------------------------------------

gd = MyMiniBatchGD(
    learning_rate=0.001
)

beale_gd_history, beale_gd_loss = run_beale(
    gd,
    "Mini-Batch Gradient Descent"
)


# ------------------------------------------------------------
# Momentum
# ------------------------------------------------------------

momentum = MyMomentum(
    learning_rate=0.001,
    momentum=0.9
)

beale_momentum_history, beale_momentum_loss = run_beale(
    momentum,
    "SGD with Momentum"
)


# ------------------------------------------------------------
# Nesterov
# ------------------------------------------------------------

nesterov = MyNesterov(
    learning_rate=0.001,
    momentum=0.9
)

beale_nesterov_history, beale_nesterov_loss = run_beale(
    nesterov,
    "Nesterov Accelerated Gradient"
)


# ------------------------------------------------------------
# Adam
# ------------------------------------------------------------

adam = MyAdam(
    learning_rate=0.01
)

beale_adam_history, beale_adam_loss = run_beale(
    adam,
    "Adam"
)
