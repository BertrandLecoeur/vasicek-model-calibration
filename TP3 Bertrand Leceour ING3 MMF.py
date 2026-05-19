import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
############################## Partie I ##############################
# Paramètres
gamma = 0.25
eta   = 0.25 * 0.03
sigma = 0.02

courbes = [
    ("Courbe croissante (r0 = 0.01)", 0.01),
    ("Courbe légèrement bossue (r0 = 0.027)", 0.027),
    ("Courbe inversée (r0 = 0.05)", 0.05),
]

# Fonctions du modèle de Vasicek
def B(t, T, gamma):
    tau = np.array(T) - t
    return (1.0 - np.exp(-gamma * tau)) / gamma

def A(t, T, gamma, eta, sigma):
    tau = np.array(T) - t
    Bb  = B(t, T, gamma)
    b   = eta / gamma
    return (b - (sigma**2)/(2.0 * gamma**2)) * (Bb - tau) - (sigma**2 * Bb**2) / (4.0 * gamma)

def yield_curve(r0, T_grid):
    t = 0.0
    A_vals = A(t, T_grid, gamma, eta, sigma)
    B_vals = B(t, T_grid, gamma)
    return (-A_vals + B_vals * r0) / T_grid

T_min = 0.05   # éviter T=0 pour pas diviser par 0
T_max = 30.0
T_grid = np.linspace(T_min, T_max, 600)

# Tracé des courbes
plt.figure(figsize=(8,5))
for nom, r0 in courbes:
    Y_vals = yield_curve(r0, T_grid)
    plt.plot(T_grid, Y_vals, label=nom)
plt.xlabel("Maturité T (années)")
plt.ylabel("Taux Y(0,T)")
plt.title("Courbes de taux du modèle de Vasicek")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tracé des courbes
plt.figure(figsize=(8,5))
Y_vals = yield_curve(0.027, T_grid)
plt.plot(T_grid, Y_vals, label="Courbe légèrement bossue (r0 = 0.027)")
plt.xlabel("Maturité T (années)")
plt.ylabel("Taux Y(0,T)")
plt.title("Courbes de taux du modèle de Vasicek")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Vérification des limites
def limite(r0):
    T_petit = 1e-6
    Y_petit = yield_curve(r0, T_petit)

    # T -> +inf: grand T
    T_grand = 2000
    Y_grand = yield_curve(r0, T_grand)

    # Limites théoriques
    limite_T0   = r0
    limite_Tinf = (eta / gamma) - 0.5 * (sigma / gamma) ** 2

    print(f"Quand r0={r0:0.3f}: "
          f"Y(T->0) = {Y_petit:.6f} et theorique={limite_T0:.6f} || "
          f"Y(T->inf) = {Y_grand:.6f} et theorique={limite_Tinf:.6f}")

print("\n--- Vérification des limites ---")
for _, r0 in courbes:
    limite(r0)

t_vals = np.linspace(0, 29.9, 100)
r_vals = np.linspace(0, 1, 100)
T_const = 30 # maturité fixée

T_mesh, R_mesh = np.meshgrid(t_vals, r_vals)

# Calcul des A(t,30) et B(t,30)
A_vals = A(T_mesh, T_const, gamma, eta, sigma)
B_vals = B(T_mesh, T_const, gamma)

# Prix du zéro-coupon
P = np.exp(A_vals - R_mesh * B_vals)

# Tracé 3D
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T_mesh, R_mesh, P, cmap='viridis', edgecolor='none')
ax.set_xlabel('Temps t')
ax.set_ylabel('Taux instantané r')
ax.set_zlabel('Prix du zéro-coupon P(t,T=30,r)')
ax.set_title('Surface du prix des obligations — P(t,T=30,r)')
plt.tight_layout()
plt.show()

Y = - (A_vals - R_mesh * B_vals) / (T_const - T_mesh)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T_mesh, R_mesh, Y, cmap='plasma', edgecolor='none')
ax.set_xlabel('Temps t')
ax.set_ylabel('Taux instantané r')
ax.set_zlabel('Taux zéro-coupon Y(t,T=30,r)')
ax.set_title('Surface du taux zéro-coupon — Y(t,T=30,r)')
plt.tight_layout()
plt.show()

############################## Partie II et III ##############################

#Levenberg Marquardt pour la calibration de Vasicek

sigma=1
eta=1
gamma=1
 
def fct_B(t, T, Gamma):
        return (1 - np.exp(-Gamma * (T - t))) / Gamma
 
def fct_A(t, T, Eta, Sigma, Gamma):
        return (fct_B(t,T,Gamma)-(T-t))*(Eta*Gamma-Sigma/2)/Gamma**2-Sigma*(fct_B(t,T,Gamma))**2/(4*Gamma)
    
#def Y(t, T, r0, eta, sigma, gamma):
    #return (-fct_A(t, T, eta, sigma, gamma) - r0 * fct_B(t, T, gamma)) / (T - t)
 
 
epsilon=10^-9
Y_M = np.array([0.035, 0.041, 0.0439, 0.046, 0.0484, 0.0494, 0.0507, 0.0514, 0.052, 0.0523])
T=np.linspace(3,30,10)

t=0
lambd=0.01
epsilon=1e-9
beta = np.array([eta, sigma**2, gamma])
r0=0.023

J=np.zeros((10,3))
d=np.array([1,1,1])

k=0

Res=np.zeros(10)
Yt=np.zeros(10)

while np.linalg.norm(d)> epsilon :
    for i in range (10):
        B=fct_B(0,T[i], beta[2])
        A=fct_A(0,T[i],beta[0],beta[1],beta[2])
        dB=((T[i]-t)*np.exp(-beta[2]*(T[i]-t))-B)/beta[2]
        dA = (beta[0] * (dB * beta[2] - B) + (T[i] - t) * beta[0]- (beta[1] / 2) * (dB - 2 * B / beta[2])- (T[i] - t) * beta[1] / beta[2]    - (beta[1] * B / 4) * (2 * beta[2] * dB - B)) / (beta[2] ** 2)
        Yt[i]= -(A - r0 * B) / (T[i] - t)
        Res[i]=Y_M[i]-Yt[i]
        J[i][0]= (B-(T[i]-t))/((T[i]-t)*beta[2])
        J[i][1]= -(1/((T[i]-t)*beta[2]))*(((B-(T[i]-t))/2*beta[2])+B**2/4)
        J[i][2]= (dA - r0 * dB) / (T[i] - t)

    d= -np.linalg.inv(J.T @ J + lambd * np.eye(3)) @ J.T @ Res
    beta= beta + d
    k=k+1

print(f"nombre d'itérations:{k}")
print(f"eta    = {beta[0]:.6f}")
print(f"sigma^2 = {beta[1]:.6f}")
print(f"gamma  = {beta[2]:.6f}")

# Plot fit
plt.plot(T, Y_M, "*", label="Yield marché (t=0)")
plt.plot(T, Yt,   "r-", label="Yield Vasicek (t=0)")
plt.xlabel("Maturité T (années)")
plt.ylabel("Taux Y(t0, T)")
plt.title("Calibration Vasicek")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#### Recalibration à t=1 et rt = 0.04 ####

sigma = 1
eta   = 1
gamma = 1
r_t   = 0.04    # taux court observé à t=1

# Données marché à t = 1
Y_M_1 = np.array([0.056, 0.064, 0.074, 0.081, 0.082, 0.090, 0.087, 0.092, 0.0895, 0.091])
T     = np.linspace(3, 30, 10)

t = 1.0
lambd   = 0.01
epsilon = 1e-9
beta    = np.array([eta, sigma**2, gamma])   # [eta, sigma^2, gamma]

J  = np.zeros((10, 3))
d  = np.array([1.0, 1.0, 1.0])
k  = 0

Res = np.zeros(10)
Yt  = np.zeros(10)

while np.linalg.norm(d) > epsilon:
    for i in range(10):
        tau = T[i] - t
        B   = fct_B(t, T[i], beta[2])
        A   = fct_A(t, T[i], beta[0], beta[1], beta[2])
        dB = (tau * np.exp(-beta[2] * tau) - B) / beta[2]
        dA = (beta[0] * (dB * beta[2] - B)+ tau * beta[0] - (beta[1] / 2) * (dB - 2 * B / beta[2]) - tau * beta[1] / beta[2] - (beta[1] * B / 4) * (2 * beta[2] * dB - B)) / (beta[2] ** 2)
        # Yield modèle à t = 1 (même signe que ton code)
        Yt[i]  = -(A - r_t * B) / tau
        Res[i] = Y_M_1[i] - Yt[i]

        # Jacobien (mêmes formules, écriture parenthésée sur J[:,1])
        J[i][0] = (B - tau) / (tau * beta[2])
        J[i][1] = - ( ((B - tau) / (2.0 * beta[2]) + (B**2) / 4.0) / (tau * beta[2]) )
        J[i][2] = (dA - r_t * dB) / tau

    d     = -np.linalg.inv(J.T @ J + lambd * np.eye(3)) @ (J.T @ Res)
    beta  = beta + d
    k    += 1

print("\n### Recalibration (t = 1) — Résultats (Q4) ###")
print(f"nombre d'itérations:{k}")
print(f"eta    = {beta[0]:.6f}")
print(f"sigma^2 = {beta[1]:.6f}")
print(f"gamma  = {beta[2]:.6f}") 

# Plot fit
plt.plot(T, Y_M_1, "*", label="Yield marché (t=1)")
plt.plot(T, Yt,   "r-", label="Yield Vasicek ajusté (t=1)")
plt.xlabel("Maturité T (années)")
plt.ylabel("Taux Y(t=1, T)")
plt.title("Recalibration Vasicek")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


#### Calibration Yield Curve LIBOR ####

# Taux en pourcentage
Y_percent = np.array([
    2.58, 2.55, 2.58, 2.52, 2.46,
    2.34, 2.46, 2.79, 3.07, 3.31,
    3.52, 3.71, 3.88, 4.02, 4.14,
    4.23, 4.33, 4.40, 4.47, 4.54,
    4.73, 4.85, 4.86
])

# Conversion en années
T_years = np.array([
    2/365,    # 2 jours 
    7/365,    # 1 semaine
    1/12,     # 1 mois
    2/12,     # 2 mois
    3/12,     # 3 mois
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    20, 25, 30
], dtype=float)

# Taux en décimal
Y_M = Y_percent / 100.0

T = T_years
n = len(T)

def fct_B(t, T, Gamma):
    return (1 - np.exp(-Gamma * (T - t))) / Gamma

def fct_A(t, T, Eta, Sigma, Gamma):
    return (fct_B(t,T,Gamma)-(T-t))*(Eta*Gamma-Sigma/2)/Gamma**2 \
           - Sigma*(fct_B(t,T,Gamma))**2/(4*Gamma)

sigma = 1
eta   = 1
gamma = 1

t      = 0.0
r0     = 0.025
lambd  = 0.01
epsilon = 1e-9

beta = np.array([eta, sigma**2, gamma], dtype=float)  # [eta, sigma^2, gamma]

J   = np.zeros((n,3))
d   = np.array([1.0,1.0,1.0])
Res = np.zeros(n)
Yt  = np.zeros(n)
k   = 0

while np.linalg.norm(d) > epsilon:
    for i in range(n):
        B = fct_B(0, T[i], beta[2])
        A = fct_A(0, T[i], beta[0], beta[1], beta[2])

        dB = ((T[i]-t)*np.exp(-beta[2]*(T[i]-t)) - B) / beta[2]
        dA = (beta[0] * (dB * beta[2] - B)
              + (T[i] - t) * beta[0]
              - (beta[1] / 2) * (dB - 2 * B / beta[2])
              - (T[i] - t) * beta[1] / beta[2]
              - (beta[1] * B / 4) * (2 * beta[2] * dB - B)) / (beta[2] ** 2)

        Yt[i]  = -(A - r0 * B) / (T[i] - t)
        Res[i] = Y_M[i] - Yt[i]

        J[i][0] = (B - (T[i]-t)) / ((T[i]-t)*beta[2])
        J[i][1] = -(1/((T[i]-t)*beta[2])) * (((B-(T[i]-t))/(2*beta[2])) + B**2/4)
        J[i][2] = (dA - r0 * dB) / (T[i] - t)

    d    = -np.linalg.inv(J.T @ J + lambd * np.eye(3)) @ (J.T @ Res)
    beta = beta + d
    k   += 1

print("### Résultats calibration t=0 sur la courbe LIBOR ###")
print(f"Nombre d'itérations : {k}")
print(f"eta    = {beta[0]:.6f}")
print(f"sigma^2 = {beta[1]:.6f}")
print(f"gamma  = {beta[2]:.6f}")

plt.figure(figsize=(8,5))
plt.plot(T, Y_M, "b*", label="Yield marché (LIBOR 7-May-2003)")
plt.plot(T, Yt, "r-", label="Yield Vasicek calibré (t=0)")
plt.xlabel("Maturité T (années)")
plt.ylabel("Taux Y(0,T)")
plt.title("Calibration Vasicek sur la courbe LIBOR (t = 0, r0 = 0.025)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#### Calibration Yield Curve Indonesian Government Securities ####

Y_M = np.array([0.0675, 0.0496, 0.0537, 0.0595, 0.0634,0.0658, 0.0735, 0.0803, 0.0847, 0.0871, 0.0882])
T = np.array([1/365,1, 2, 3, 4, 5, 10, 15, 20, 25, 30], dtype=float)


n = len(T)

sigma = 1
eta   = 1
gamma = 1

t      = 0.0
r0     = 0.0675
lambd  = 0.01
epsilon = 1e-9

beta = np.array([eta, sigma**2, gamma], dtype=float)  # [eta, sigma^2, gamma]

J   = np.zeros((n,3))
d   = np.array([1.0,1.0,1.0])
Res = np.zeros(n)
Yt  = np.zeros(n)
k   = 0

while np.linalg.norm(d) > epsilon:
    for i in range(n):
        B = fct_B(0, T[i], beta[2])
        A = fct_A(0, T[i], beta[0], beta[1], beta[2])

        dB = ((T[i]-t)*np.exp(-beta[2]*(T[i]-t)) - B) / beta[2]
        dA = (beta[0] * (dB * beta[2] - B)
              + (T[i] - t) * beta[0]
              - (beta[1] / 2) * (dB - 2 * B / beta[2])
              - (T[i] - t) * beta[1] / beta[2]
              - (beta[1] * B / 4) * (2 * beta[2] * dB - B)) / (beta[2] ** 2)

        Yt[i]  = -(A - r0 * B) / (T[i] - t)
        Res[i] = Y_M[i] - Yt[i]

        J[i][0] = (B - (T[i]-t)) / ((T[i]-t)*beta[2])
        J[i][1] = -(1/((T[i]-t)*beta[2])) * (((B-(T[i]-t))/(2*beta[2])) + B**2/4)
        J[i][2] = (dA - r0 * dB) / (T[i] - t)

    d    = -np.linalg.inv(J.T @ J + lambd * np.eye(3)) @ (J.T @ Res)
    beta = beta + d
    k   += 1

print("### Résultats calibration t=0 sur la courbe Indonesian Government Securities ###")
print(f"Nombre d'itérations : {k}")
print(f"eta    = {beta[0]:.6f}")
print(f"sigma^2 = {beta[1]:.6f}")
print(f"gamma  = {beta[2]:.6f}")

plt.figure(figsize=(8,5))
plt.plot(T, Y_M, "b*", label="Yield marché (Indonesian Government Securities 22 Jul)")
plt.plot(T, Yt, "r-", label="Yield Vasicek calibré (t=0)")
plt.xlabel("Maturité T (années)")
plt.ylabel("Taux Y(0,T)")
plt.title("Calibration Vasicek sur la courbe Indonesian Government Securities (t = 0, r0 = 0.0675)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

############################## Partie IV ##############################
T = 5              # horizon en années
N = 200            # nombre de pas
dt = T / N

eta   = 0.6       # niveau de long terme
gamma = 4.0        # vitesse de réversion
sigma = 0.08       # volatilité
r0    = 0.03       # taux initial

r = np.zeros(N+1)
r[0] = r0

for i in range(N):
    Z = np.random.normal()
    r[i+1] = (
        r[i] * np.exp(-gamma * dt)
        + eta * (1 - np.exp(-gamma * dt)) / gamma
        + sigma * np.sqrt((1 - np.exp(-2 * gamma * dt)) / (2 * gamma)) * Z
    )

t = np.linspace(0, T, N+1)

plt.figure(figsize=(8,4))
plt.plot(t, r, label="r(t) simulé (Vasicek)")
plt.xlabel("Temps (années)")
plt.ylabel("Taux court r(t)")
plt.title("Simulation d’un processus de Vasicek (Question 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

x = r[:-1]
y = r[1:]

plt.figure(figsize=(6,5))
plt.scatter(x, y, s=10, alpha=0.6)
plt.xlabel("$x_i = r_i$")
plt.ylabel("$y_i = r_{i+1}$")
plt.title("Relation entre r_i et r_{i+1} (Vasicek)")
plt.grid(True)
plt.tight_layout()
plt.show()

def f(a, b, x):
    return a * x + b

# méthode analytique
a_ols, b_ols = np.polyfit(x, y, 1)


# Levenberg-Marquardt
M = len(x)
a = 0.5   # point de depart
b = 0.1
lam = 0.01
eps = 1e-10

for k in range(50):
    y_pred = f(a, b, x)
    res = y - y_pred

    J = np.zeros((M, 2))
    J[:, 0] = x
    J[:, 1] = 1.0
    H = J.T @ J + lam * np.eye(2)
    g = J.T @ res
    delta = np.linalg.inv(H) @ g
    a_new = a + delta[0]
    b_new = b + delta[1]

    # Acceptation si la nouvelle erreur baisse
    if np.sum((y - f(a_new, b_new, x))**2) < np.sum(res**2):
        a, b = a_new, b_new
        lam /= 10
    else:
        lam *= 10

    # Arrêt si petit
    if np.linalg.norm(delta) < eps:
        break

# Calcul de la variance D^2
D2 = np.mean((y - (a*x + b))**2)

print("### Résultats Question 2 ###")
print(f"a (LM) = {a:.6f}")
print(f"b (LM) = {b:.6f}")
print(f"a (methode analytique) = {a_ols:.6f}")
print(f"b (methode analytique) = {b_ols:.6f}")
print(f"Variance D^2 = {D2:.8f}")

plt.figure(figsize=(6,5))
plt.scatter(x, y, s=10, alpha=0.6, label="Points (r_i, r_{i+1})")
plt.plot(x, a*x + b, 'r', label=f"LM : y = {a:.4f} x + {b:.4f}")
plt.xlabel("$r_i$")
plt.ylabel("$r_{i+1}$")
plt.title("Ajustement LM : r_{i+1} = a r_i + b")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

################# vitesse de convergence #################
def LM_regression(x, y, a0, b0, lam, max_iter=50, eps=1e-10):
    a, b = a0, b0
    M = len(x)

    for k in range(max_iter):
        y_pred = a*x + b
        res = y - y_pred

        J = np.column_stack([x, np.ones(M)])
        H = J.T @ J + lam * np.eye(2)
        g = J.T @ res
        delta = np.linalg.inv(H) @ g

        a_new = a + delta[0]
        b_new = b + delta[1]

        # Acceptation
        if np.sum((y - (a_new*x + b_new))**2) < np.sum(res**2):
            a, b = a_new, b_new
            lam /= 10
        else:
            lam *= 10

        if np.linalg.norm(delta) < eps:
            return a, b, k+1

    return a, b, max_iter

# Tests demandés par le TP
lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
points_initial = [(0.5, 0.1), (1.5, -0.5), (-0.5, 1.0)]

print("### Étude de convergence LM ###")
for lam in lambdas:
    for (a0, b0) in points_initial:
        a_hat, b_hat, iters = LM_regression(x, y, a0, b0, lam)
        print(f"λ = {lam:>6}, init=({a0:>4},{b0:>4}) -> a={a_hat:.6f}, b={b_hat:.6f}, iters={iters}")
        
################# fin vitesse de convergence #################

# Estimation de gamma
gamma_hat = -np.log(a) / dt

# Estimation de eta
eta_hat = b * gamma_hat / (1 - a)

# Estimation de sigma
sigma_hat = np.sqrt(D2 * (2 * gamma_hat) / (1 - a**2))

print("### Estimation des paramètres Vasicek (Question 3) ###")
print(f"gamma_hat = {gamma_hat:.6f}")
print(f"eta_hat   = {eta_hat:.6f}")
print(f"sigma_hat = {sigma_hat:.6f}")


#### Données réelles Federal Funds Effective Rate ####

df = pd.read_csv("DFF.csv")
df.iloc[:,0] = pd.to_datetime(df.iloc[:,0]) # dates
r = df["DFF"].values # taux courts (jours)

x = r[:-1]
y = r[1:]

dt = 1/365


plt.figure(figsize=(6,5))
plt.scatter(x, y, s=10, alpha=0.6)
plt.xlabel("$x_i = r_i$")
plt.ylabel("$y_i = r_{i+1}$")
plt.title("Relation entre r_i et r_{i+1} (Données FRED)")
plt.grid(True)
plt.tight_layout()
plt.show()

a_ols, b_ols = np.polyfit(x, y, 1)

def f(a,b,x):
    return a*x + b

M = len(x)
a = 0.5
b = 0.1
lam = 0.01
eps = 1e-10

for k in range(50):
    y_pred = f(a,b,x)
    res = y - y_pred

    J = np.zeros((M,2))
    J[:,0] = x
    J[:,1] = 1.0

    H = J.T @ J + lam*np.eye(2)
    g = J.T @ res

    delta = np.linalg.inv(H) @ g

    a_new = a + delta[0]
    b_new = b + delta[1]

    # accepter si meilleure erreur
    if np.sum((y - f(a_new,b_new,x))**2) < np.sum(res**2):
        a, b = a_new, b_new
        lam /= 10
    else:
        lam *= 10

    if np.linalg.norm(delta) < eps:
        break

# variance
D2 = np.mean((y - (a*x + b))**2)

print("### Résultats Régression ###")
print(f"a (LM) = {a:.6f}")
print(f"b (LM) = {b:.6f}")
print(f"a (Méthode analytique) = {a_ols:.6f}")
print(f"b (Méthode analytique) = {b_ols:.6f}")
print(f"Variance D^2 = {D2:.8f}")

plt.figure(figsize=(6,5))
plt.scatter(x, y, s=10, label="FRED")
plt.plot(x, a*x + b, "r", label=f"LM : y = {a:.4f}x + {b:.4f}")
plt.legend()
plt.grid(True)
plt.show()

gamma_hat = -np.log(a) / dt
eta_hat   = b / (1 - a)
sigma_hat = np.sqrt(D2 * 2*gamma_hat / (1 - np.exp(-2*gamma_hat*dt)))

print("\n### Paramètres Vasicek (historique FRED) ###")
print(f"gamma_hat = {gamma_hat:.6f}")
print(f"eta_hat   = {eta_hat:.6f}")
print(f"sigma_hat = {sigma_hat:.6f}")


