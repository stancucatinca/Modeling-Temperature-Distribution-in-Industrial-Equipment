import numpy as np
import matplotlib.pyplot as plt


def create_grid(nx, ny, Lx, Ly):
    """
    Creează o grilă discretizată pentru domeniu.
    :param nx: Numărul de puncte pe axa x.
    :param ny: Numărul de puncte pe axa y.
    :param Lx: Lungimea domeniului pe axa x.
    :param Ly: Lățimea domeniului pe axa y.
    :return: Vectorii coordonatelor (x, y) și pasul discretizării (dx, dy).
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    return x, y, dx, dy


def initialize_temperature(nx, ny, boundary_conditions):
    """
    Inițializează câmpul de temperatură și aplică condițiile de frontieră.
    :param nx: Numărul de puncte pe axa x.
    :param ny: Numărul de puncte pe axa y.
    :param boundary_conditions: Temperaturi la margini sub formă de dicționar.
    :return: Matricea temperaturii inițiale.
    """
    T = np.zeros((ny, nx))
    T[0, :] = boundary_conditions['top']
    T[-1, :] = boundary_conditions['bottom']
    T[:, 0] = boundary_conditions['left']
    T[:, -1] = boundary_conditions['right']
    return T


def solve_laplace(T, dx, dy, max_iter=1000, tol=1e-6):
    """
    Rezolvă ecuația lui Laplace pentru distribuția temperaturii.
    :param T: Matricea temperaturii inițiale.
    :param dx: Pasul discretizării pe axa x.
    :param dy: Pasul discretizării pe axa y.
    :param max_iter: Numărul maxim de iterații.
    :param tol: Toleranța pentru convergență.
    :return: Matricea temperaturii la convergență.
    """
    ny, nx = T.shape
    for it in range(max_iter):
        T_new = T.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                T_new[i, j] = (
                                      (T[i + 1, j] + T[i - 1, j]) / dx ** 2 +
                                      (T[i, j + 1] + T[i, j - 1]) / dy ** 2
                              ) / (2 / dx ** 2 + 2 / dy ** 2)
        if np.max(np.abs(T_new - T)) < tol:
            print(f"Converged after {it + 1} iterations.")
            break
        T = T_new
    return T


def solve_poisson(T, dx, dy, source, k, c_p, max_iter=1000, tol=1e-6):
    """
    Rezolvă ecuația lui Poisson pentru distribuția temperaturii.
    :param T: Matricea temperaturii inițiale.
    :param dx: Pasul discretizării pe axa x.
    :param dy: Pasul discretizării pe axa y.
    :param source: Matricea sursei de căldură.
    :param k: Conductivitatea termică.
    :param c_p: Capacitatea termică.
    :param max_iter: Numărul maxim de iterații.
    :param tol: Toleranța pentru convergență.
    :return: Matricea temperaturii la convergență.
    """
    ny, nx = T.shape
    alpha = k / c_p  # Difuzivitatea termică (m²/s)
    for it in range(max_iter):
        T_new = T.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                # Formula discretizată pentru Poisson cu sursă
                T_new[i, j] = (
                                      (T[i + 1, j] + T[i - 1, j]) / dx ** 2 +
                                      (T[i, j + 1] + T[i, j - 1]) / dy ** 2 -
                                      source[i, j] / alpha  # Include sursa de căldură
                              ) / (2 / dx ** 2 + 2 / dy ** 2)

        # Verificarea convergenței
        if np.max(np.abs(T_new - T)) < tol:
            print(f"Converged after {it + 1} iterations.")
            break
        T = T_new
    return T


def plot_temperature(x, y, T, title):
    """
    Generează un grafic al distribuției temperaturii.
    :param x: Vectorul coordonatelor pe axa x.
    :param y: Vectorul coordonatelor pe axa y.
    :param T: Matricea temperaturii.
    :param title: Titlul graficului.
    """
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, T, 50, cmap='hot')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()


if __name__ == "__main__":
    Lx = float(input("Dimensiune axa x (lungimea domeniului, metri): "))
    Ly = float(input("Dimensiune axa y (lățimea domeniului, metri): "))
    nx = int(input("Numărul de puncte pe axa x: "))
    ny = int(input("Numărul de puncte pe axa y: "))

    boundary_conditions = {
        'top': float(input("Temperatura margine de jos (°C): ")),
        'bottom': float(input("Temperatura margine de sus (°C): ")),
        'left': float(input("Temperatura margine stânga (°C): ")),
        'right': float(input("Temperatura margine dreapta (°C): "))
    }

    k = float(input("Conductivitatea termică (W/mK): "))
    c_p = float(input("Capacitatea termică (J/kgK): "))

    x, y, dx, dy = create_grid(nx, ny, Lx, Ly)

    T = initialize_temperature(nx, ny, boundary_conditions)

    # Definirea sursei de căldură
    source = np.zeros((ny, nx))
    intensity = float(input("Intensitatea sursei de căldură (W/m³): "))
    source[ny // 3:2 * ny // 3, nx // 3:2 * nx // 3] = intensity  # Zonă centrală

    print("Rezolvă ecuația lui Laplace...")
    T_laplace = solve_laplace(T.copy(), dx, dy)
    plot_temperature(x, y, T_laplace, "Distribuția temperaturii - Ecuația lui Laplace")

    print("Rezolvă ecuația lui Poisson...")
    T_poisson = solve_poisson(T.copy(), dx, dy, source, k, c_p)
    plot_temperature(x, y, T_poisson, "Distribuția temperaturii - Ecuația lui Poisson")
