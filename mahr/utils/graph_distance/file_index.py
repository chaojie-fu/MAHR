import os

import pandas as pd


def generate_file_index(asf_amc_root_dir, index_file, motion_category="example"):
    df = pd.read_json(index_file, dtype={"file name": str})

    # generate and save file list
    file_index = []

    if motion_category == "example":
        for index, row in df.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            if subject == "01" and motion_index == "01":
                file_index.append({
                    "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                    "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                    "fps": row["frame rate"],
                    "description": row["file description"],
                    "tag": "all"
                })

    elif motion_category == "all":
        for index, row in df.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            file_index.append({
                "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                "fps": row["frame rate"],
                "description": row["file description"],
                "tag": "all"
            })

    elif motion_category == "walk":
        # locate all walk motions
        plane_ground_walk_filter = df[(df["file description"].str.contains("walk").fillna(False)) &
                                      ~(df["file description"].str.contains("terrain").fillna(False)) &
                                      ~(df["file description"].str.contains("paint").fillna(False)) &
                                      ~(df["file description"].str.contains("subjects").fillna(False)) &
                                      ~(df["file description"].str.contains("vignettes").fillna(False)) &
                                      ~(df["file description"].str.contains("object").fillna(False)) &
                                      ~(df["file description"].str.contains("attack").fillna(False)) &
                                      ~(df["file description"].str.contains("hair").fillna(False)) &
                                      ~(df["file description"].str.contains("stairs").fillna(False)) &
                                      ~(df["file description"].str.contains("ladder").fillna(False)) &
                                      ~(df["file description"].str.contains("wipe").fillna(False)) &
                                      ~(df["file description"].str.contains("stretching").fillna(False)) &
                                      ~(df["file description"].str.contains("basketball").fillna(False)) &
                                      ~(df["file description"].str.contains("sit").fillna(False)) &
                                      ~(df["file description"].str.contains("jump").fillna(False)) &
                                      ~(df["file description"].str.contains("ball").fillna(False))]

        for index, row in plane_ground_walk_filter.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            file_index.append({
                "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                "fps": row["frame rate"],
                "description": row["file description"],
                "tag": "walk"
            })

    elif motion_category == "run":
        # locate all running motions
        run_filter = df[(df["file description"].str.contains("run").fillna(False)) &
                        ~(df["file description"].str.contains("terrain").fillna(False)) &
                        ~(df["file description"].str.contains("subjects").fillna(False)) &
                        ~(df["file description"].str.contains("vignettes").fillna(False)) &
                        ~(df["file description"].str.contains("jump").fillna(False)) &
                        ~(df["file description"].str.contains("sit").fillna(False)) &
                        ~(df["file description"].str.contains("drunk").fillna(False)) &
                        ~(df["file description"].str.contains("Drunk").fillna(False)) &
                        ~(df["file description"].str.contains("walk").fillna(False))
                        ]

        for index, row in run_filter.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            file_index.append({
                "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                "fps": row["frame rate"],
                "description": row["file description"],
                "tag": "run"
            })
    elif motion_category == "dance":
        # locate all dancing motions
        dance_filter = df[(df["file description"].str.contains("dance").fillna(False)) &
                          ~(df["file description"].str.contains("terrain").fillna(False))]
        for index, row in dance_filter.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            file_index.append({
                "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                "fps": row["frame rate"],
                "description": row["file description"],
                "tag": "dance"
            })

        file_index = []
        for index, row in df.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            file_index.append({
                "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                "fps": row["frame rate"],
                "description": row["file description"],
                "tag": "dance"
            })

    elif motion_category == "running":
        # locate all running motions
        custom_filter = df[(df["file name"].str.contains("^02_03").fillna(False)) |
                           (df["file name"].str.contains("^16_41").fillna(False)) |
                           (df["file name"].str.contains("^16_44").fillna(False)) |
                           (df["file name"].str.contains("^02_05").fillna(False)) |
                           (df["file name"].str.contains("^13_29").fillna(False)) |
                           (df["file name"].str.contains("^02_06").fillna(False)) |
                           (df["file name"].str.contains("^02_07").fillna(False)) |
                           (df["file name"].str.contains("^05_*").fillna(False)) |
                           (df["file name"].str.contains("^85_*").fillna(False)) |
                           (df["file name"].str.contains("^90_*").fillna(False)) |
                           (df["file name"].str.contains("^60_02").fillna(False)) |
                           (df["file name"].str.contains("^61_13").fillna(False)) |
                           (df["file name"].str.contains("^08_07").fillna(False)) |
                           (df["file name"].str.contains("^14_03").fillna(False)) |
                           (df["file name"].str.contains("^16_17").fillna(False)) |
                           (df["file name"].str.contains("^16_19").fillna(False)) |
                           (df["file name"].str.contains("^40_09").fillna(False)) |
                           (df["file name"].str.contains("^41_02").fillna(False)) |
                           (df["file name"].str.contains("^13_04").fillna(False)) |
                           (df["file name"].str.contains("^82_05").fillna(False)) |
                           (df["file name"].str.contains("^13_19").fillna(False)) |
                           (df["file name"].str.contains("^49_06").fillna(False)) |
                           (df["file name"].str.contains("^87_01").fillna(False)) |
                           (df["file name"].str.contains("^87_03").fillna(False)) |
                           (df["file name"].str.contains("^79_02").fillna(False)) |
                           (df["file name"].str.contains("^140_*").fillna(False))
                           ]

        for index, row in custom_filter.iterrows():
            file_name = row["file name"]
            subject, motion_index = file_name.split("_")
            file_index.append({
                "asf": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}.asf"),
                "amc": os.path.join(asf_amc_root_dir, f"{subject}", f"{subject}_{motion_index}.amc"),
                "fps": row["frame rate"],
                "description": row["file description"],
                "tag": "robot range test"
            })
    else:
        raise ValueError(f"Undefined motion category {motion_category}. "
                         f"Motion Category should be one of: all, walk, run, dance, running")
    return file_index
