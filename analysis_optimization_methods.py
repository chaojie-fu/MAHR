import os
import numpy as np
import sqlite3
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# Centralized font size configuration
FONT_SCALE_ENV = float(os.getenv("MAHR_FONT_SCALE", "1.4"))
FONT_SIZES = {
    "title": 14 * FONT_SCALE_ENV,
    "axes_label": 12 * FONT_SCALE_ENV,
    "ticks": 10 * FONT_SCALE_ENV,
    "legend": 10 * FONT_SCALE_ENV,
}


# Optionally push some defaults into matplotlib so any future text uses these sizes
matplotlib.rcParams.update({
    "axes.titlesize": FONT_SIZES["title"],
    "axes.labelsize": FONT_SIZES["axes_label"],
    "xtick.labelsize": FONT_SIZES["ticks"],
    "ytick.labelsize": FONT_SIZES["ticks"],
    "legend.fontsize": FONT_SIZES["legend"],
})


def comparing_global_with_local_optimization(db_path, db_continuous_path):
    conn_db = sqlite3.connect(db_path, timeout=30.0)
    conn_db_continuous = sqlite3.connect(db_continuous_path, timeout=30.0)

    plot_figure_1 = True
    plot_figure_2 = True
    plot_figure_3 = True

    # ============================================================

    optimizer_unique_query = """
                             SELECT DISTINCT optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol
                             FROM optimization_runs \
                             """
    df_optimizer_unique = pd.read_sql_query(optimizer_unique_query, conn_db)

    unique_opptimizer_names_mapping = {}
    for index, row in df_optimizer_unique.iterrows():
        optimizer = row['optimizer']
        lbfgsb_maxiter = row['lbfgsb_maxiter']
        de_maxiter = row['de_maxiter']
        de_popsize = row['de_popsize']
        de_strategy = row['de_strategy']
        de_tol = row['de_tol']

        if optimizer == 'lbfgsb':
            # unique_name = f"L-BFGS-B\nMaxiter={lbfgsb_maxiter}"
            unique_name = f"L-BFGS-B"
        elif optimizer == 'de':
            unique_name = f"Differential Evolution\nMaxiter={de_maxiter}"
        else:
            raise ValueError(f"Unknown optimizer {optimizer}")

        unique_opptimizer_names_mapping[
            (optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol)] = unique_name

    # scatter for visualizing convergence of different optimization methods
    # first, get unique bvh_file_name, frame_index, norm_pose from conn_db
    problem_unique_query = """
                           SELECT DISTINCT bvh_file_name, frame_index, norm_pose
                           FROM optimization_runs \
                           """
    df_problem_unique = pd.read_sql_query(problem_unique_query, conn_db)

    os.makedirs('comparing_optimizations', exist_ok=True)


    # ============================================================
    if plot_figure_1:
        # initialize a dataframe to store results, which should have two columns, 'optimizer' and 'global_optimality_gap'
        results1 = pd.DataFrame(columns=['optimizer', 'global_optimality_gap'])
        # calculate the optimal value for each unique problem
        for index, row in df_problem_unique.iterrows():
            bvh_file_name = row['bvh_file_name']
            frame_index = row['frame_index']
            norm_pose = row['norm_pose']

            # get data from conn_db
            query_db = f"""
            SELECT optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol, v0_raw, v0_penalized, opt_raw, opt_penalized
            FROM optimization_runs 
            WHERE bvh_file_name = '{bvh_file_name}' 
            AND frame_index = {frame_index} 
            AND norm_pose = '{norm_pose}'
            """
            df_db = pd.read_sql_query(query_db, conn_db)

            # find the minimum opt_penalized value in df_db as global optimum
            global_optimum = df_db['opt_penalized'].min()

            # append the 'optimizer' and 'global_optimality_gap' to results1 dataframe
            for idx, r in df_db.iterrows():
                optimizer = r['optimizer']
                lbfgsb_maxiter = r['lbfgsb_maxiter']
                de_maxiter = r['de_maxiter']
                de_popsize = r['de_popsize']
                de_strategy = r['de_strategy']
                de_tol = r['de_tol']
                opt_penalized = r['opt_penalized']

                if optimizer == "lbfgsb" and lbfgsb_maxiter != 5000:
                    continue

                unique_name = unique_opptimizer_names_mapping[(optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol)]
                global_optimality_gap = opt_penalized - global_optimum

                # add entry to the dataframe
                results1 = pd.concat(
                    [
                        results1,
                        pd.DataFrame(
                            {
                                'optimizer': [unique_name],
                                'global_optimality_gap': [global_optimality_gap]
                             }
                        )
                    ],
                    ignore_index=True)

        # plot the scatter plot, with y axis as categorical optimizer, x axis as global_optimality_gap
        # use log scale for x axis, jitter every point for its label (optimizer) a bit for better visualization
        cat = pd.Categorical(results1['optimizer'],
                             categories=reversed([
                                 "L-BFGS-B",
                                 "Differential Evolution\nMaxiter=50",
                                 "Differential Evolution\nMaxiter=100",
                                 "Differential Evolution\nMaxiter=500",
                                 "Differential Evolution\nMaxiter=1000",
                                 "Differential Evolution\nMaxiter=2000",
                                 "Differential Evolution\nMaxiter=5000",
                                 "Differential Evolution\nMaxiter=10000"
                             ]),
                             ordered=True)
        y_numeric = cat.codes.astype(float)

        rng = np.random.default_rng(0)
        y_jittered = y_numeric + rng.normal(loc=0.0, scale=0.12, size=len(y_numeric))

        x_vals = results1['global_optimality_gap'].to_numpy()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.scatter(x_vals, y_jittered, alpha=0.6, s=16, edgecolor='none')
        # additionally, plot the mean line for each optimizer
        points = []
        for i, category in enumerate(cat.categories):
            mask = results1['optimizer'] == category
            median_gap = np.median(results1.loc[mask, 'global_optimality_gap'])
            points.append([median_gap, i])
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], 'ko-', markersize=8, alpha=0.6, label='Median Global Optimality Gap')

        ax.set_xscale('log')
        ax.set_yticks(ticks=np.arange(len(cat.categories)), labels=cat.categories)
        ax.set_ylim(-0.5, len(cat.categories) - 0.5)
        ax.set_xlabel('Global Optimality Gap', fontsize=FONT_SIZES["axes_label"])
        # ax.set_ylabel('Optimizer', fontsize=FONT_SIZES["axes_label"])
        ax.set_title('Global Optimality Gap Across Optimizers', fontsize=FONT_SIZES["title"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["ticks"])
        ax.grid(True, which="both", ls="--", linewidth=0.5, axis='x')
        fig.tight_layout()

        fig.savefig('comparing_optimizations/global_optimality_gap_by_optimizer.png', dpi=300)
        plt.show()
    # ============================================================

    # ============================================================
    if plot_figure_2:
        # initialize a dataframe to store results, which should have two columns, 'optimizer' and 'global_optimality_gap'
        results2 = pd.DataFrame(columns=['optimizer', 'initial_value', 'optimized_value'])
        # calculate the optimal value for each unique problem
        for index, row in df_problem_unique.iterrows():
            bvh_file_name = row['bvh_file_name']
            frame_index = row['frame_index']
            norm_pose = row['norm_pose']

            # get data from conn_db
            query_db = f"""
            SELECT optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol, v0_raw, v0_penalized, opt_raw, opt_penalized
            FROM optimization_runs 
            WHERE bvh_file_name = '{bvh_file_name}' 
            AND frame_index = {frame_index} 
            AND norm_pose = '{norm_pose}'
            """
            df_db = pd.read_sql_query(query_db, conn_db)

            # find the minimum opt_penalized value in df_db as global optimum
            global_optimum = df_db['opt_penalized'].min()

            # append the 'optimizer' and 'global_optimality_gap' to results dataframe
            for idx, r in df_db.iterrows():
                optimizer = r['optimizer']
                v0_penalized = r['v0_penalized']
                lbfgsb_maxiter = r['lbfgsb_maxiter']
                de_maxiter = r['de_maxiter']
                de_popsize = r['de_popsize']
                de_strategy = r['de_strategy']
                de_tol = r['de_tol']
                opt_penalized = r['opt_penalized']

                if optimizer == "lbfgsb" and lbfgsb_maxiter != 5000:
                    continue

                unique_name = unique_opptimizer_names_mapping[(optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol)]

                # add entry to the dataframe
                results2 = pd.concat(
                    [
                        results2,
                        pd.DataFrame(
                            {
                                'optimizer': [unique_name],
                                'initial_value': [v0_penalized],
                                'optimized_value': [opt_penalized]
                            }
                        )
                    ],
                    ignore_index=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        cats = pd.Categorical(results2['optimizer'],
                              categories=[
                                  "L-BFGS-B",
                                  "Differential Evolution\nMaxiter=50",
                                  "Differential Evolution\nMaxiter=100",
                                  "Differential Evolution\nMaxiter=500",
                                  "Differential Evolution\nMaxiter=1000",
                                  "Differential Evolution\nMaxiter=2000",
                                  "Differential Evolution\nMaxiter=5000",
                                  "Differential Evolution\nMaxiter=10000"
                              ],
                              ordered=True)
        # use a colormap to assign colors to each category
        colors = plt.cm.viridis(cats.codes / cats.categories.size)

        for cat in cats.categories:
            mask = results2['optimizer'] == cat
            ax.scatter(
                results2.loc[mask, 'initial_value'],
                results2.loc[mask, 'optimized_value'],
                color=colors[mask][0],
                s=40, label=cat, alpha=0.6, edgecolor='none')

        # plot scatter with x as initial_value and y as optimized_value, then hue by optimizer
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.set_xlabel("Initial Value", fontsize=FONT_SIZES["axes_label"])
        ax.set_ylabel("Optimized Value", fontsize=FONT_SIZES["axes_label"])
        ax.set_title("Initial vs Optimized Value Across Optimizers", fontsize=FONT_SIZES["title"])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["ticks"])
        ax.grid(True, which="major", ls="--", linewidth=0.5, axis='both')
        fig.tight_layout()

        fig.savefig('comparing_optimizations/initial_value_to_optimized_value.png', dpi=300)
        plt.show()
    # ============================================================

    # ============================================================
    if plot_figure_3:
        # initialize a dataframe to store results, which should have two columns, 'optimizer' and 'global_optimality_gap'
        results3 = pd.DataFrame(columns=['optimized_value', 'runtime_sec'])
        # calculate the optimal value for each unique problem
        for index, row in df_problem_unique.iterrows():
            bvh_file_name = row['bvh_file_name']
            frame_index = row['frame_index']
            norm_pose = row['norm_pose']

            # get data from conn_db
            query_db = f"""
            SELECT optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol, v0_raw, v0_penalized, opt_raw, opt_penalized, runtime_sec
            FROM optimization_runs 
            WHERE bvh_file_name = '{bvh_file_name}' 
            AND frame_index = {frame_index} 
            AND norm_pose = '{norm_pose}'
            """
            df_db = pd.read_sql_query(query_db, conn_db)

            # find the minimum opt_penalized value in df_db as global optimum
            global_optimum = df_db['opt_penalized'].min()

            # append the 'optimizer' and 'optimized_value' to results dataframe
            for idx, r in df_db.iterrows():
                optimizer = r['optimizer']
                lbfgsb_maxiter = r['lbfgsb_maxiter']
                de_maxiter = r['de_maxiter']
                de_popsize = r['de_popsize']
                de_strategy = r['de_strategy']
                de_tol = r['de_tol']
                opt_penalized = r['opt_penalized']
                runtime_sec = r['runtime_sec']

                if optimizer == "lbfgsb":
                    continue

                unique_name = unique_opptimizer_names_mapping[(optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol)]

                # add entry to the dataframe
                results3 = pd.concat(
                    [
                        results3,
                        pd.DataFrame(
                            {
                                'optimizer': [unique_name],
                                'optimized_value': [opt_penalized],
                                'runtime_sec': [runtime_sec],
                            }
                        )
                    ],
                    ignore_index=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        cats = pd.Categorical(results3['optimizer'],
                              categories=[
                                  "Differential Evolution\nMaxiter=50",
                                  "Differential Evolution\nMaxiter=100",
                                  "Differential Evolution\nMaxiter=500",
                                  "Differential Evolution\nMaxiter=1000",
                                  "Differential Evolution\nMaxiter=2000",
                                  "Differential Evolution\nMaxiter=5000",
                                  "Differential Evolution\nMaxiter=10000"
                              ],
                              ordered=True)
        # use a colormap to assign colors to each category
        colors = plt.cm.viridis(cats.codes / cats.categories.size)

        for cat in cats.categories:
            mask = results3['optimizer'] == cat
            ax.scatter(
                results3.loc[mask, 'runtime_sec'],
                results3.loc[mask, 'optimized_value'],
                color=colors[mask][0],
                s=40, label=cat, alpha=0.6, edgecolor='none')
        ax.set_yscale('log')
        ax.set_xlabel("Optimization Times (s)", fontsize=FONT_SIZES["axes_label"])
        ax.set_ylabel("Optimized Values", fontsize=FONT_SIZES["axes_label"])
        ax.set_title("Time–Accuracy Trade-off in Differential Evolution Optimizers", fontsize=FONT_SIZES["title"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["ticks"])
        ax.grid(True, which="both", ls="--", linewidth=0.5, axis='both')
        fig.tight_layout()
        fig.savefig('comparing_optimizations/runtime_wrt_optimized_values.png', dpi=300)
        plt.show()
    # ============================================================


def comparing_interframe_with_global_optimization(db_path, db_continuous_path):
    conn_db = sqlite3.connect(db_path, timeout=30.0)
    conn_db_continuous = sqlite3.connect(db_continuous_path, timeout=30.0)

    optimizer_unique_query = """
    SELECT DISTINCT optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol
    FROM optimization_runs_continuous \
    """
    df_optimizer_unique = pd.read_sql_query(optimizer_unique_query, conn_db_continuous)

    unique_opptimizer_names_mapping = {}
    for index, row in df_optimizer_unique.iterrows():
        optimizer = row['optimizer']
        lbfgsb_maxiter = row['lbfgsb_maxiter']
        de_maxiter = row['de_maxiter']
        de_popsize = row['de_popsize']
        de_strategy = row['de_strategy']
        de_tol = row['de_tol']

        if optimizer == 'lbfgsb':
            # unique_name = f"L-BFGS-B\nMaxiter={lbfgsb_maxiter}"
            unique_name = f"L-BFGS-B"
        elif optimizer == 'de':
            unique_name = f"Differential Evolution\nMaxiter={de_maxiter}"
        else:
            raise ValueError(f"Unknown optimizer {optimizer}")

        unique_opptimizer_names_mapping[
            (optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol)] = unique_name

    # scatter for visualizing convergence of different optimization methods
    # first, get unique bvh_file_name, frame_index, norm_pose from conn_db_continuous
    problem_unique_query = """
                           SELECT DISTINCT bvh_file_name, frame_index, norm_pose
                           FROM optimization_runs_continuous \
                           """
    df_problem_unique = pd.read_sql_query(problem_unique_query, conn_db_continuous)

    os.makedirs('comparing_optimizations', exist_ok=True)

    # initialize a dataframe to store results, which should have two columns, 'optimizer' and 'opt_penalized'
    results = pd.DataFrame(columns=['optimizer', 'opt_penalized', 'frame_index'])

    # iterate over unique problems
    for index, row in df_problem_unique.iterrows():
        bvh_file_name = row['bvh_file_name']
        frame_index = row['frame_index']
        norm_pose = row['norm_pose']

        # get data from conn_db_continuous
        query_db = f"""
        SELECT optimizer, frame_index, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol, v0_raw, v0_penalized, opt_raw, opt_penalized
        FROM optimization_runs_continuous
        WHERE bvh_file_name = '{bvh_file_name}' 
        AND frame_index = {frame_index} 
        AND norm_pose = '{norm_pose}'
        """
        df_db = pd.read_sql_query(query_db, conn_db_continuous)

        # append the 'optimizer' and 'opt_penalized' and 'frame_index' to results1 dataframe
        for idx, r in df_db.iterrows():
            optimizer = r['optimizer']
            lbfgsb_maxiter = r['lbfgsb_maxiter']
            de_maxiter = r['de_maxiter']
            de_popsize = r['de_popsize']
            de_strategy = r['de_strategy']
            de_tol = r['de_tol']
            opt_penalized = r['opt_penalized']
            frame_index = r['frame_index']

            if optimizer == "lbfgsb" and lbfgsb_maxiter != 5000:
                continue

            if optimizer == "de" and de_maxiter not in [500, 2000, 10000]:
                continue

            unique_name = unique_opptimizer_names_mapping[
                (optimizer, lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol)]

            # add entry to the dataframe
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            'optimizer': [unique_name],
                            'opt_penalized': [opt_penalized],
                            'frame_index': [frame_index]
                        }
                    )
                ],
                ignore_index=True)

    # plot line plot with x as frame_index and y as opt_penalized, hue by optimizer
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cats = pd.Categorical(results['optimizer'],
                          categories=[
                              "L-BFGS-B",
                              # "Differential Evolution\nMaxiter=50",
                              # "Differential Evolution\nMaxiter=100",
                              "Differential Evolution\nMaxiter=500",
                              # "Differential Evolution\nMaxiter=1000",
                              "Differential Evolution\nMaxiter=2000",
                              # "Differential Evolution\nMaxiter=5000",
                              "Differential Evolution\nMaxiter=10000"
                          ],
                          ordered=True)
    # use a colormap to assign colors to each category
    colors = plt.cm.viridis(cats.codes / cats.categories.size)
    for cat in cats.categories:
        mask = results['optimizer'] == cat
        # sort by frame_index then plot lines
        df_plot = results.loc[mask].sort_values(by='frame_index')
        ax.plot(
            df_plot['frame_index'],
            df_plot['opt_penalized'],
            color=colors[mask][0],
            label=cat, alpha=0.6)
    ax.set_xlabel("Frame", fontsize=FONT_SIZES["axes_label"])
    ax.set_ylabel("Optimized Value", fontsize=FONT_SIZES["axes_label"])
    interval = (df_plot['opt_penalized'].max() - df_plot['opt_penalized'].min())
    ax.set_ylim(df_plot['opt_penalized'].min() - 0.1 * interval,
                df_plot['opt_penalized'].max() + 0.3 * interval)
    ax.set_title("Optimized Value For A Single Mocap Trajectory Across Optimizers", fontsize=FONT_SIZES["title"])
    # resize the legend to make it four-columns
    # place it outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, fontsize=FONT_SIZES["legend"])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["ticks"])
    ax.grid(True, which="both", ls="--", linewidth=0.5, axis='both')
    fig.tight_layout()
    fig.savefig('comparing_optimizations/interframe_optimization.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    db = "log/optimizations.sqlite"
    db_continuous = "log/optimizations_continuous.sqlite"

    comparing_global_with_local_optimization(db, db_continuous)
    comparing_interframe_with_global_optimization(db, db_continuous)
