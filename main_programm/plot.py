import numpy as np
import matplotlib.pyplot as plt

def plot_prod_index_conlengths(
    path: str,
    prod_per: np.array,
    con_lengths: np.array,
    links_per: np.array,
    energy_loss_per: np.array,
    supply_per: np.array
    ) -> None:
    """
    asd
    """
    prod_per = prod_per * 100

    fig, ax = plt.subplots(3, 1, figsize=(8, 13.5), sharex=True)
    ax1, ax2, ax3 = ax

    for i in range(len(con_lengths)):
        ax1.plot(
            prod_per,
            links_per[i,:],
            marker="o",
            markersize=3,
            label=f"Radius = {round(con_lengths[i], 2)}m")
        ax2.plot(
            prod_per,
            energy_loss_per[i,:],
            marker="o",
            markersize=3,
            label=f"Radius = {round(con_lengths[i], 2)}m")
        ax3.plot(
            prod_per,
            supply_per[i,:],
            marker="o",
            markersize=3,
            label=f"Radius = {round(con_lengths[i], 2)}m")


    ax1.grid(color = "#595959", linestyle='dotted', lw = 0.7, alpha = 0.8)
    ax1.set_xlabel("Percentage of producers")
    ax1.set_ylabel("Links Percentage")
    ax1.legend(shadow=True)

    ax2.grid(color = "#595959", linestyle='dotted', lw = 0.7, alpha = 0.8)
    ax2.set_xlabel("Percentage of producers")
    ax2.set_ylabel("Energy Loss Percentage")
    ax2.legend(shadow=True)

    ax3.grid(color = "#595959", linestyle='dotted', lw = 0.7, alpha = 0.8)
    ax3.set_xlabel("Percentage of producers")
    ax3.set_ylabel("Supply Percentage")
    ax3.set_xticks(np.linspace(10, 100, 10))
    ax3.legend(shadow=True)
    plt.savefig(path, bbox_inches='tight')

def plot_res_index_cap_res(
    path: str,
    res_per: np.array,
    cap_res: np.array,
    links_per: np.array,
    energy_loss_per: np.array,
    supply_per: np.array,
    links_per_std: np.array,
    energy_loss_per_std: np.array,
    supply_per_std: np.array
    ) -> None:
    """
    asd
    """
    res_per = res_per * 100
    
    size = 8
    fig, ax = plt.subplots(3, 1, figsize=(size, size*1.6))
    ax1, ax2, ax3 = ax

    for i in range(len(cap_res)):
        ax1.errorbar(
            res_per,
            links_per[i,:],
            yerr = links_per_std[i, :],
            marker="o",
            markersize=3,
            label=f"Reservoir = {round(cap_res[i], 2)}kWh")
        ax2.errorbar(
            res_per,
            energy_loss_per[i,:],
            yerr = energy_loss_per_std[i, :],
            marker="o",
            markersize=3,
            label=f"Reservoir = {round(cap_res[i], 2)}kWh")
        ax3.errorbar(
            res_per,
            supply_per[i,:],
            yerr = supply_per_std[i, :],
            marker="o",
            markersize=3,
            label=f"Reservoir = {round(cap_res[i], 2)}kWh")


    ax1.grid(color = "#595959", linestyle='dotted', lw = 0.7, alpha = 0.8)
    ax1.set_xlabel("Reservoir percentage")
    ax1.set_ylabel("Links Percentage")
    ax1.set_xticks(np.linspace(0, 100, 11))
    ax1.legend(shadow=True)

    ax2.grid(color = "#595959", linestyle='dotted', lw = 0.7, alpha = 0.8)
    ax2.set_xlabel("Reservoir percentage")
    ax2.set_ylabel("Energy Loss Percentage")
    ax2.set_xticks(np.linspace(0, 100, 11))
    ax2.legend(shadow=True)

    ax3.grid(color = "#595959", linestyle='dotted', lw = 0.7, alpha = 0.8)
    ax3.set_xlabel("Reservoir percentage")
    ax3.set_ylabel("Supply Percentage")
    ax3.set_xticks(np.linspace(0, 100, 11))
    ax3.legend(shadow=True)
    plt.savefig(path, bbox_inches='tight')