# Uebungsblatt3_MEM26_Hilfsfunktionen.py
#
# Hilfsfunktionen für Übungsblatt 3 in "Mechanistische Effektmodellierung" 2026.
# Die Datei muss im selben Ordner liegen wie das Notebook. Sie müssen hier nichts ändern.
#
# Inhalt:
# - Allgemein: Klasse ModelFit (Kalibrierung mit scipy.optimize.minimize, Nelder-Mead),
#   Verlustfunktion logMSE, calc_S_max (maximale strukturelle Masse), Einlesen der Daten
# - Aufgabe 2 (DEB-Kalibrierung an Kontrolldaten): load_control_data, plot_control_data,
#   define_defaultparams_control (Startwerte), setup_modelfit, fit_model
# - Aufgabe 3 (DEB-TKTD-Kalibrierung, Azoxystrobin): load_tox_data, plot_tox_data,
#   define_defaultparams_tox (fixierte DEB-Parameter), fit_debtox


import pandas as pd
import numpy as np
import os
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from mempyDEB.DEBODE.simulators import simulate_DEBBase, constant_exposures
from mempyDEB.DEBODE.defaultparams import Params

# Konstanten

DATA_DIR = 'data' # Ordner mit den Daten (aus der zip-Datei)
EMB_DEV_TIME = 2 # geschätzte Embryonalentwicklungszeit (realistisch für Daphnia)
S_MAX_REFERENCE = 388.4 # maximale strukturelle Masse, die durch die Anfangswerte impliziert ist
EXPOSURES = [0, 2, 10, 40, 100, 150, 200, 250] # getestete Konzentrationen im Azoxystrobin-Experiment


# ==================== Allgemeine Funktionen ====================

class ModelFit:
    """
    Sammelt alles, was für eine Kalibrierung gebraucht wird: Daten, Simulator, Verlustfunktion,
    Anfangswerte (`intguess`) und alle übrigen Parameter (`defaultparams`).
    `run_optimization()` ruft intern `scipy.optimize.minimize()` auf.
    """

    def __init__(self):

        self.data = None
        self.simulator = None
        self.loss = None
        self.intguess: dict = None
        self.defaultparams = None

        self.optimization_result = None

    def define_objective_function(self):

        def objective_function(parvals):
            """
            Zielfunktion: simuliert das Modell mit den vorgeschlagenen Parameterwerten
            und gibt den Wert der Verlustfunktion zurück.
            """

            # alle kalibrierten Parameter müssen positiv sein, sonst wird der Vorschlag verworfen
            if np.any(np.array(parvals) <= 0):
                return np.inf

            pfit = self.intguess.copy()
            pfit.update(dict(zip(self.intguess.keys(), parvals)))

            prediction = self.simulator(pfit)

            return self.loss(prediction, self.data)

        return objective_function

    def run_optimization(self, method = 'Nelder-Mead', **kwargs):
        """
        Kalibrierung mit `scipy.optimize.minimize`.
        """

        objective_function = self.define_objective_function()

        opt = minimize(
            objective_function, # Zielfunktion
            list(self.intguess.values()), # Anfangswerte
            method = method, # Optimierungsverfahren
            **kwargs
            )

        self.optimization_result = opt
        self.p_opt = dict(zip(self.intguess.keys(), opt.x))


def logMSE(D, P):
    """
    Mittlere quadratische Abweichung der log-transformierten Werte.
    """
    return np.sum(((np.log(D + 1) - np.log(P + 1))**2)/len(D))


def calc_S_max(spc: dict):
    """
    Berechnung der maximalen strukturellen Masse auf Basis der DEB-Parameter.
    """
    return np.power((spc['kappa'] * spc['Idot_max_rel'] * spc['eta_IA_0']) / spc['k_M_0'], 3)


def read_azoxy_data():
    """
    Einlesen der Wachstums- und Reproduktionsdaten für D. magna (alle Konzentrationen).
    """

    azoxy_growth = pd.read_csv(os.path.join(DATA_DIR, 'azoxy_static_growth_tidy.csv')).drop(['Unnamed: 0', 'length_mm'], axis = 1) # Wachstumsdaten
    azoxy_repro = pd.read_csv(os.path.join(DATA_DIR, 'azoxy_static_repro_tidy.csv'), encoding = 'utf-8-sig').rename(columns = {'cum_repro' : 'cum_repro_mean'}) # Reproduktionsdaten

    # für Wachstum haben wir Rohdaten (Beobachtung pro Replikat), für Reproduktion nur Mittelwerte
    # wir berechnen daher für Wachstum ebenfalls den Mittelwert (und der Vollständigkeit halber Standardabweichung und Anzahl)
    azoxy_growth_agg = azoxy_growth.groupby(['C_W', 't_day']).S.agg(S_mean = 'mean', S_sd = 'std', n_S = 'count').reset_index()

    # Zusammenfügen der Datensätze
    data = pd.merge(azoxy_growth_agg, azoxy_repro, on = ['t_day', 'C_W'], how = 'outer').fillna(0).assign(
        S_mean = lambda df : df.S_mean.replace({0 : np.nan})
        )

    return data


def relative_response(df, S = 'S', R = 'cum_repro'):
    """
    Berechnet die relativen Antworten y_S = S/S_Kontrolle und y_R = R/R_Kontrolle für jeden Zeitpunkt.
    Ist die Reproduktion in der Kontrolle noch 0, wird y_R = 1 gesetzt.
    """

    controls = df.loc[df['C_W'] == 0, ['t_day', S, R]]
    df = pd.merge(df, controls, on = ['t_day'], suffixes = ['', '_ref'])

    df['y_S'] = df[S] / df[S + '_ref']
    df['y_R'] = (df[R] / df[R + '_ref'].replace({0 : np.nan})).fillna(1.)

    return df


# ==================== Aufgabe 2: DEB-Kalibrierung an Kontrolldaten ====================

def load_control_data():
    """
    Einlesen der Kontrolldaten (ohne Chemikalie) für D. magna.
    """

    data = read_azoxy_data()
    data = data.loc[lambda df : df.C_W == 0].drop('C_W', axis = 1).reset_index(drop = True) # wir benötigen nur die Kontrolle

    return data


def plot_control_data(data):
    """
    Plotten der Kontrolldaten für D. magna.
    Gibt ein `fig, ax`-Tuple zurück.
    """

    fig, ax = plt.subplots(ncols = 2, figsize = (8,3.5), sharex = True)

    sns.lineplot(data, x = 't_day', y = 'S_mean', ax = ax[0], marker = 'o', color = 'black', label = 'Daten')
    sns.lineplot(data, x = 't_day', y = 'cum_repro_mean', ax = ax[1], marker = 'o', color = 'black')

    ax[0].set(xlabel = "Zeit (d)", ylabel = "Strukturelle Masse (mugC)")
    ax[1].set(xlabel = "Zeit (d)", ylabel = "Kumulative Reproduktion (#)")

    sns.despine()
    plt.tight_layout()

    return fig,ax


def define_defaultparams_control():
    """
    Standard-Parameterwerte für die DEB-Kalibrierung (D. magna, S_max = 388.4 mugC).
    Idot_max_rel, Idot_max_rel_emb, eta_IA_0, k_M_0, S_p und X_emb_int sind das Ergebnis aus A1e
    (Musterlösung), damit Aufgabe 2 nicht vom eigenen Ergebnis aus Aufgabe 1 abhängt.
    Modellwährung ist mugC.
    """

    glb = {
        'C_W': 0.0,
        'V_patch': 0.05,
        'Xdot_in': 3596.296296296296,
        'a_int': 2,
        'tspan': (0, 25)
    }

    spc = {
        'X_emb_int': 19.42,
        'eta_IA_0': 0.3333333333333333,
        'eta_AS_0': 0.9,
        'eta_AR_0': 0.95,
        'Idot_max_rel': 16.21364127123944,
        'Idot_max_rel_emb': 12.256744759847304,
        'K_X': 500.0,
        'kappa': 0.9,
        'eta_SA': 0.9,
        'k_M_0': 0.6666666666666666,
        'S_p': 258.93333333333334,
        'kD_j': 0.5,
        'ED50_j': 1.0,
        'beta_j': 2.0,
        'pmoa': 'G',
        'kD_h': 0.5,
        'ED50_h': 2.0,
        'beta_h': 1.0
        }

    return Params(glb = glb, spc = spc)


def define_simulator_control(f: ModelFit):
    """
    Simulator-Funktion für die DEB-Kalibrierung.
    Ändert sich durch die Parameter die maximale Größe S_max, werden Idot_max_rel_emb, X_emb_int
    und S_p mit dem Zoom-Faktor mitskaliert.
    """

    def simulator(theta: dict):

        p = deepcopy(f.defaultparams)
        p.spc.update(theta) # macht das gleiche wie p.spc['Idot_max_rel'] = theta['Idot_max_rel'] für alle Einträge in theta

        S_max_theta = calc_S_max(p.spc) # implizierte maximale Struktur auf Basis von theta
        zoom_factor_theta = S_max_theta / S_MAX_REFERENCE # Zoom-Faktor auf Basis von theta

        # diese Parameter skalieren mit der neuen maximalen Größe
        p.spc['Idot_max_rel_emb'] *= zoom_factor_theta**(1/3)
        p.spc['X_emb_int'] *= zoom_factor_theta
        if 'S_p' not in f.intguess: # wird S_p selbst kalibriert, wird es nicht mitskaliert
            p.spc['S_p'] *= zoom_factor_theta

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = simulate_DEBBase(p).assign(
                cum_repro = lambda df : np.trunc(df.R / p.spc['X_emb_int']).shift(EMB_DEV_TIME, fill_value = 0)
                ).rename({'t' : 't_day'}, axis = 1)

        return prediction

    return simulator


def define_loss_control(constants):

    def loss(prediction: pd.DataFrame, data: pd.DataFrame) -> float:

        # Skalierung der Vorhersagen
        prediction = prediction.assign(
            S_scaled = lambda df : df.S / constants['scale_factor_S'],
            cum_repro_scaled = lambda df : df.cum_repro / constants['scale_factor_R']
            )

        # Zusammenfassung von Vorhersagen und Daten in einen einzelnen Datensatz
        eval_df = pd.merge(prediction, data, on = 't_day', how = 'right', suffixes = ['_predicted', '_observed'])

        # ungültige Simulation (NaN) wird verworfen
        if eval_df[['S_scaled_predicted', 'cum_repro_scaled_predicted']].isna().values.any():
            return np.inf

        # Berechnung der Verlustfunktion
        loss_S = logMSE(eval_df.S_scaled_predicted, eval_df.S_scaled_observed)
        loss_R = logMSE(eval_df.cum_repro_scaled_predicted, eval_df.cum_repro_scaled_observed)

        return loss_S + loss_R

    return loss


def setup_modelfit():

    f = ModelFit()
    f.data = load_control_data()

    # Konstanten die während des Fittings genutzt werden

    constants = {
        'scale_factor_S' : np.max(f.data.S_mean),
        'scale_factor_R' : np.max(f.data.cum_repro_mean)
    }

    # Skalierung der Daten

    f.data = f.data.assign(
        S_scaled = lambda df : df.S_mean / constants['scale_factor_S'], # Berechnung der skalierten Struktur
        cum_repro_scaled = lambda df : df.cum_repro_mean / constants['scale_factor_R'] # Berechnung der skalierten Reproduktion
        )

    # Definition von Anfangswerten der Parameter

    f.defaultparams = define_defaultparams_control() # enthält alle Parameter, die für die Simulationen notwendig sind

    # enthält nur die Parameter, die kalibriert werden
    # was nicht in intguess ist, wird auf den Wert in defaultparams fixiert

    f.intguess = {
        'Idot_max_rel' : f.defaultparams.spc['Idot_max_rel']
        }

    f.simulator = define_simulator_control(f)
    f.loss = define_loss_control(constants)

    return f


def fit_model():

    f = setup_modelfit() # generiere ModelFit-Instanz
    f.run_optimization() # führe Kalibrierung durch

    print(f"Geschätzte Parameterwerte: {f.p_opt}")

    # Simulation optimierter Parameter

    p = deepcopy(f.defaultparams)
    p.spc.update(f.p_opt)

    sim_opt = f.simulator(p.spc)

    # Visual predictive check

    fig, ax = plot_control_data(f.data)

    sns.lineplot(sim_opt, x = 't_day', y = 'S', ax = ax[0], label = "Modell")
    sns.lineplot(sim_opt, x = 't_day', y = 'cum_repro', ax = ax[1])
    plt.show()

    return f


# ==================== Aufgabe 3: DEB-TKTD-Kalibrierung (Azoxystrobin) ====================

def load_tox_data():
    """
    Einlesen der Daten für alle Konzentrationen und Berechnung der relativen Antworten y_S und y_R.
    """

    data = read_azoxy_data()

    return relative_response(data, S = 'S_mean', R = 'cum_repro_mean')


def plot_tox_data(data):
    """
    Plotten der relativen Antworten für alle Konzentrationen.
    Gibt ein `fig, ax`-Tuple zurück.
    """

    # Plot-Matrix mit 8 Spalten für Konzentrationen und 2 Zeilen für Wachstum und Reproduktion
    fig, ax = plt.subplots(ncols = 8, nrows = 2, figsize = (20,8), sharey = True)

    for (i,C_W) in enumerate(np.sort(data.C_W.unique())):

        ax[0,i].set(title = f'{C_W}')
        obs = data.loc[lambda df : df.C_W==C_W]

        sns.lineplot(obs, x = 't_day', y = 'y_S', ax = ax[0,i], marker = 'o', color = 'black')
        sns.lineplot(obs, x = 't_day', y = 'y_R', ax = ax[1,i], marker = 'o', color = 'black', label = "Daten")

    [a.legend().remove() for a in np.ravel(ax) if a.get_legend()]
    ax[1,0].legend()
    ax[0,0].set_ylim(0, 2.)
    ax[0,0].set(ylabel = "Relative Struktur")
    ax[1,0].set(ylabel = "Relative Reproduktion")

    sns.despine()
    plt.tight_layout()

    return fig,ax


def plot_sim(ax, sim):

    for (i,C_W) in enumerate(np.sort(sim.C_W.unique())):

        df = sim.loc[lambda df : df.C_W == C_W]

        sns.lineplot(df, x = 't_day', y = 'y_S', ax = ax[0,i])
        sns.lineplot(df, x = 't_day', y = 'y_R', ax = ax[1,i])

    return ax


def define_defaultparams_tox():
    """
    Standard-Parameterwerte für die DEB-TKTD-Kalibrierung.
    Die DEB-Parameter sind auf Werte fixiert, die an die Kontrolldaten kalibriert wurden.
    """

    glb = {
        'C_W': 0.,
        'V_patch': 0.05,
        'Xdot_in': 3596.296296296296,
        'a_int': 2,
        'tspan': (0, 25)
    }

    spc = {
        'X_emb_int': 19.42,
        'eta_IA_0': 0.3333333333333333,
        'eta_AS_0': 0.9,
        'eta_AR_0': 0.95,
        'Idot_max_rel': 12.256744759847304,
        'Idot_max_rel_emb': 12.256744759847304,
        'K_X': 500.0,
        'kappa': 0.9,
        'eta_SA': 0.9,
        'k_M_0': 0.5039684199579493,
        'S_p': 258.93333333333334,
        'kD_j': 0.5,
        'ED50_j': 1.0,
        'beta_j': 2.0,
        'pmoa': 'G', # PMoA wird in fit_debtox überschrieben!
        'kD_h': 0.5,
        'ED50_h': 2.0,
        'beta_h': 1.0
        }

    # geschätzte DEB-Parameter aus der vorherigen Übung (MEM25)
    fitted_params =  {
        'Idot_max_rel': 22.88376226064429,
        'kappa': 0.5265137793254286,
        'eta_AS_0': 0.6767736358373964,
        'S_p': 121.62673237331643
        }

    spc.update(fitted_params)

    return Params(glb = glb, spc = spc)


def define_simulator_tox(f: ModelFit):
    """
    Simulator-Funktion für die DEB-TKTD-Kalibrierung: simuliert alle Konzentrationen
    und berechnet die relativen Antworten.
    """

    def simulator(theta: dict):

        p = deepcopy(f.defaultparams)
        p.spc.update(theta)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = constant_exposures(
                simulate_DEBBase, p, EXPOSURES
                ).assign(
                    cum_repro = lambda df : np.trunc(df.R / p.spc['X_emb_int']).shift(EMB_DEV_TIME, fill_value = 0)
                ).rename({'t' : 't_day'}, axis = 1)

            return relative_response(prediction)

    return simulator


def define_loss_tox():

    def loss(prediction: pd.DataFrame, data: pd.DataFrame) -> float:

        # Zusammenfassung von Vorhersagen und Daten in einen einzelnen Datensatz
        eval_df = pd.merge(prediction, data, on = ['t_day', 'C_W'], how = 'right', suffixes = ['_predicted', '_observed'])

        # ungültige Simulation (NaN) wird verworfen
        if eval_df[['y_S_predicted', 'y_R_predicted']].isna().values.any():
            return np.inf

        # Berechnung der Verlustfunktion
        loss_S = logMSE(eval_df.y_S_predicted, eval_df.y_S_observed)
        loss_R = logMSE(eval_df.y_R_predicted, eval_df.y_R_observed)

        return loss_S + loss_R

    return loss


def fit_debtox(data, pmoa = 'G'):
    """
    Kalibriert die TKTD-Parameter kD_j, ED50_j und beta_j des DEB-TKTD-Modells für eine PMoA.

    Parameters
    ----------
    data: pandas.DataFrame
        Daten aus load_tox_data()
    pmoa: str
        Physiologische Wirkungsweise: 'G', 'M', 'A' oder 'R'

    Returns
    ----------
    f: ModelFit
        Enthält u.a. die geschätzten Parameter (f.p_opt) und das Ergebnis der Optimierung (f.optimization_result)
    """

    f = ModelFit()
    f.data = data
    f.defaultparams = define_defaultparams_tox()
    f.defaultparams.spc['pmoa'] = pmoa

    f.intguess = {
        'kD_j' : 1.,
        'ED50_j' : np.median(EXPOSURES),
        'beta_j' : 2.
        }

    f.simulator = define_simulator_tox(f)
    f.loss = define_loss_tox()

    f.run_optimization()

    print(f"PMoA {pmoa}")
    print(f"Geschätzte Parameterwerte: { {k : round(float(v), 4) for k, v in f.p_opt.items()} }")
    print(f"Wert der Verlustfunktion: {f.optimization_result.fun:.4f}")

    # Simulation mit den optimierten Parametern und Plot
    sim_opt = f.simulator(f.p_opt)
    fig, ax = plot_tox_data(data)
    plot_sim(ax, sim_opt)
    fig.suptitle(f"PMoA {pmoa}")
    plt.show()

    return f
