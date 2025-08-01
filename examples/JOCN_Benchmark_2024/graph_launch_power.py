import argparse
import logging
import os
import random
from typing import Tuple

import numpy as np
import time
from datetime import datetime
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.core.bands import BandS, BandC, BandL

# ===================================================
# 1. Definição da Função run_environment
# ===================================================
def run_environment(    
    n_eval_episodes,
    heuristic,
    monitor_file_name,
    topology,
    seed,
    allow_rejection,
    load,
    episode_length,
    num_spectrum_resources,
    launch_power_dbm,
    bandwidth,
    frequency_start,
    frequency_slot_bandwidth,
    bit_rate_selection,
    bit_rates,
    margin,
    file_name,
    measure_disruptions,
    defragmentation,
    n_defrag_services,
    gen_observation,
) -> None:
    """
    Executa o ambiente com a heurística especificada e salva os resultados em um arquivo CSV.

    :param n_eval_episodes: Número de episódios a serem executados.
    :param heuristic: Índice da heurística a ser utilizada.
    :param monitor_file_name: Nome base para o arquivo CSV de monitoramento.
    :param topology: Objeto de topologia.
    :param seed: Semente para geração de números aleatórios.
    :param allow_rejection: Permitir rejeição de solicitações.
    :param load: Carga a ser utilizada na simulação.
    :param episode_length: Número de chegadas por episódio.
    :param num_spectrum_resources: Número de recursos de espectro.
    :param launch_power_dbm: Potência de lançamento em dBm.
    :param bandwidth: Largura de banda.
    :param frequency_start: Frequência inicial.
    :param frequency_slot_bandwidth: Largura de banda do slot de frequência.
    :param bit_rate_selection: Seleção de taxa de bits.
    :param bit_rates: Taxas de bits disponíveis.
    :param margin: Margem.
    :param file_name: Nome do arquivo para salvar serviços.
    :param measure_disruptions: Medir interrupções.
    """
    from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
    from optical_networking_gym.heuristics.heuristics import (
        shortest_available_path_first_fit_best_modulation,
        shortest_available_path_lowest_spectrum_best_modulation,
        best_modulation_load_balancing,
        load_balancing_best_modulation,
        rnd,
        heuristic_shortest_available_path_first_fit_best_modulation,
        heuristic_highest_snr,
        heuristic_lowest_fragmentation,
        heuristic_mscl,
        heuristic_mscl_simplified,
        heuristic_mscl_sequential_simplified,
        #heuristic_load_balancing_first_fit,
        heuristic_psr,
    )

    # Configurações do ambiente
    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=allow_rejection,
        load=load,
        episode_length=10,
        num_spectrum_resources=num_spectrum_resources,
        launch_power_dbm=launch_power_dbm,
        bandwidth=bandwidth,
        frequency_start=frequency_start,
        frequency_slot_bandwidth=frequency_slot_bandwidth,
        bit_rate_selection=bit_rate_selection,
        bit_rates=bit_rates,
        margin=margin,
        file_name=file_name,
        measure_disruptions=measure_disruptions,
        k_paths=3, 
        modulations_to_consider=6,
        defragmentation=defragmentation,
        n_defrag_services=n_defrag_services,
        gen_observation=gen_observation,
        bands=[BandS(), BandC(), BandL()],  # SUPORTE MULTIBANDA
    )

    # Seleção da heurística baseada no índice
    fn_heuristic = 10
    if heuristic == 1:
        fn_heuristic = heuristic_shortest_available_path_first_fit_best_modulation
    elif heuristic == 2:
        fn_heuristic = shortest_available_path_lowest_spectrum_best_modulation
    elif heuristic == 3:
        fn_heuristic = best_modulation_load_balancing
    elif heuristic == 4:
        fn_heuristic = load_balancing_best_modulation
    elif heuristic == 5:
        fn_heuristic = heuristic_mscl
    elif heuristic == 6:
        fn_heuristic = heuristic_mscl_simplified
    elif heuristic == 7:
        fn_heuristic = heuristic_mscl_sequential_simplified
    #elif heuristic == 8:
    #    fn_heuristic = heuristic_load_balancing_first_fit
    elif heuristic == 9:
        print("[INFO] Rodando PSR-C (variant='C') em ambiente multibanda!")
        fn_heuristic = lambda env: heuristic_psr(env, variant='C')
    elif heuristic == 10:
        fn_heuristic = lambda env: heuristic_psr(env, variant='O')
    else:
        raise ValueError(f"Heuristic index `{heuristic}` is not found!")

    # Criação do ambiente
    env = QRMSAEnvWrapper(**env_args)
    env.reset()

    # Verificação do nome do arquivo de monitoramento
    if monitor_file_name is None:
        raise ValueError("Missing monitor file name")

    # Definição do nome final do arquivo CSV
    monitor_final_name = "_".join([
        monitor_file_name, 
        topology.name, 
        str(env.env.launch_power_dbm), 
        str(env.env.load) + ".csv"
    ])
    # Descomente a linha abaixo se desejar evitar sobrescrever arquivos existentes
    # if os.path.exists(monitor_final_name):
    #     raise ValueError(f"File `{monitor_final_name}` already exists!")
    
    # Preparação do arquivo CSV
    with open(monitor_final_name, "wt", encoding="UTF-8") as file_handler:
        file_handler.write(f"# Date: {datetime.now()}\n")
        header = (
            "episode,service_blocking_rate,episode_service_blocking_rate,"
            "bit_rate_blocking_rate,episode_bit_rate_blocking_rate"
        )
        for mf in env.env.modulations:
            header += f",modulation_{mf.spectral_efficiency}"
        header += ",episode_disrupted_services,episode_time\n"
        file_handler.write(header)

        # Loop de execução dos episódios
        for ep in range(n_eval_episodes):
            obs, info = env.reset()

            done = False
            start_time = time.time()

            while not done:
                action,_,_ = fn_heuristic(env)
                _, _, done, _, info = env.step(action)
            
            end_time = time.time()
            ep_time = end_time - start_time

            # Impressão das informações do episódio
            print(f"Episódio {ep} finalizado.")
            print(info)

            # Escrita das métricas no CSV
            row = (
                f"{ep},{info['service_blocking_rate']},"
                f"{info['episode_service_blocking_rate']},"
                f"{info['bit_rate_blocking_rate']},"
                f"{info['episode_bit_rate_blocking_rate']}"
            )
            for mf in env.env.modulations:
                row += f",{info.get(f'modulation_{mf.spectral_efficiency}', 0.0)}"
            row += f",{info.get('episode_disrupted_services', 0)},{ep_time:.2f}\n"
            file_handler.write(row)

    print(f"\nFinalizado! Resultados salvos em: {monitor_final_name}")

# ===================================================
# 2. Configuração de Logging e Semente
# ===================================================
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

seed = 20
random.seed(seed)
np.random.seed(seed)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optical Network Simulation')

    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nobel-eu.xml',
        help='Network topology file to be used (default: nsfnet_chen.txt)'
    )

    parser.add_argument(
        '-e', '--num_episodes',
        type=int,
        default=100,
        help='Number of episodes to be simulated (default: 1000)'
    )

    parser.add_argument(
        '-s', '--episode_length',
        type=int,
        default=50000,
        help='Number of arrivals per episode to be generated (default: 50)'
    )

    parser.add_argument(
        '-l', '--load',
        type=int,
        default=150,
        help='Load to be used in the simulation (default: 210)'
    )

    parser.add_argument(
        '-th', '--threads',
        type=int,
        default= 5,
        help='Number of threads to be used for running simulations (default: 2)'
    )

    # Argumentos adicionais para a heurística e nome do arquivo de monitoramento
    parser.add_argument(
        '-hi', '--heuristic_index',
        type=int,
        default=9,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Heuristic index to be used (1: First Fit, 2: Lowest Spectrum, 3: Load Balancing Modulation, 4: Load Balancing Best Modulation, 9: PSR-C, 10: PSR-O)'
    )

    parser.add_argument(
        '-mf', '--monitor_file_name',
        type=str,
        default='examples/jocn_benchmark_2024/results/simulation_updteded_th_results',
        help='Base name for the monitor CSV file'
    )

    return parser.parse_args()

# ===================================================
# 4. Função Principal
# ===================================================
def main():
    args = parse_arguments()

    # Definição das modulações (mantendo as mesmas configurações)
    cur_modulations: Tuple[Modulation] = (
        Modulation(
            name="BPSK",
            maximum_length=100_000,  # 100,000 km para garantir segurança
            spectral_efficiency=1,
            #minimum_osnr=12.6,
            minimum_osnr=3.71,
            inband_xt=-14,
        ),
        Modulation(
            name="QPSK",
            maximum_length=2_000,
            spectral_efficiency=2,
            #minimum_osnr=12.6,
            minimum_osnr=6.72,
            inband_xt=-17,
        ),
        Modulation(
            name="8QAM",
            maximum_length=1_000,
            spectral_efficiency=3,
            #minimum_osnr=18.6,
            minimum_osnr=10.84,
            inband_xt=-20,
        ),
        Modulation(
            name="16QAM",
            maximum_length=500,
            spectral_efficiency=4,
            #minimum_osnr=22.4,
            minimum_osnr=13.24,
            inband_xt=-23,
        ),
        Modulation(
            name="32QAM",
            maximum_length=250,
            spectral_efficiency=5,
            #minimum_osnr=26.4,
            minimum_osnr=16.16,
            inband_xt=-26,
        ),
        Modulation(
            name="64QAM",
            maximum_length=125,
            spectral_efficiency=6,
            # minimum_osnr=30.4,
            minimum_osnr=19.01,
            inband_xt=-29,
        ),
    )

    # Parâmetros padrão
    attenuation_db_km = 0.2
    default_noise_figure_db = 4.5

    # Carregamento da topologia
    topology_path =  r'examples/topologies/germany50.xml'#os.path.join(
    #      "examples", "topologies", args.topology_file
    #  )
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topology file '{topology_path}' not found.")

    topology = get_topology(
        topology_path,           # Path to the topology file
        None,                # Name of the topology
        cur_modulations,         # Tuple of modulation formats
        80,                      # Maximum span length in km
        attenuation_db_km,       # Default attenuation in dB/km
        default_noise_figure_db, # Default noise figure in dB
        5                        # Number of shortest paths to compute between node pairs    
        )

    # Parâmetros de simulação
    threads = args.threads
    num_spectrum_resources = 640
    frequency_slot_bandwidth = 12.5e9
    bandwidth = num_spectrum_resources * frequency_slot_bandwidth  # 8e11
    frequency_start = 3e8 / 1565e-9
    bit_rates = (10, 40, 100, 400)

    # Definição das potências de lançamento
    launch_powers = np.linspace(-8, 8, num=17)
    env_args = []

    # Preparação dos argumentos de simulação para cada potência de lançamento
    for launch_power in launch_powers:
        sim_args = (
            args.num_episodes,             # n_eval_episodes
            args.heuristic_index,          # heuristic
            args.monitor_file_name,        # monitor_file_name
            topology,                      # topology
            seed,                          # seed
            True,                          # allow_rejection
            150,                           # load
            args.episode_length,           # episode_length
            num_spectrum_resources,        # num_spectrum_resources
            launch_power,                  # launch_power_dbm
            bandwidth,                     # bandwidth
            frequency_start,               # frequency_start
            frequency_slot_bandwidth,      # frequency_slot_bandwidth
            "discrete",                    # bit_rate_selection
            bit_rates,                     # bit_rates
            0,                             # margin
            "examples/jocn_benchmark_2024/results/lp_services_1",  # file_name
            False,                          # measure_disruptions
            False,
            0,                             # n_defrag_services
            False,                         # defragmentation
        )
        env_args.append(sim_args)

    # Execução das simulações com ou sem multiprocessing baseado na contagem de threads
    if threads > 1:
        from multiprocessing import Pool  # Importação para evitar importação desnecessária
        with Pool(processes=threads) as pool:
            # Usando starmap para mapear múltiplos argumentos
            pool.starmap(run_environment, env_args)
    else:
        for arg in env_args:
            run_environment(*arg)

    print("Todas as simulações foram executadas.")

# ===================================================
# 5. Execução da Função Principal
# ===================================================
if __name__ == "__main__":
    main()