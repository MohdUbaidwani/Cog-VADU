import json
import argparse
import os


def main(Raw_stage_folder, output_raw_scores_dir, output_raw_summary_dir):
    os.makedirs(output_raw_scores_dir, exist_ok=True)
    os.makedirs(output_raw_summary_dir, exist_ok=True)

    for filename in os.listdir(Raw_stage_folder):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(Raw_stage_folder, filename)
        print(f"Processing: {input_path}")

        with open(input_path, "r") as f:
            data = json.load(f)

        raw_scores = {}
        raw_summaries = {}

        for key, value in data.items():
            if isinstance(value, dict):
                if "analysis" in value:
                    analysis = value["analysis"]

                    anomaly_score = analysis.get("anomaly_score", 0)
                    summary = analysis.get("description", "")

                    raw_scores[key] = anomaly_score
                    raw_summaries[key] = summary

                else:
                    raw_scores[key] = 0
                    raw_summaries[key] = ""

            else:
                print(
                    f"Filename {filename} -> "
                    f"Key {key} has unsupported format {type(value)}"
                )

        # Save scores
        raw_output_scores_path = os.path.join(
            output_raw_scores_dir, filename
        )

        with open(raw_output_scores_path, "w") as f:
            json.dump(raw_scores, f, indent=4)

        # Save summaries/descriptions
        raw_output_summary_path = os.path.join(
            output_raw_summary_dir, filename
        )

        with open(raw_output_summary_path, "w") as f:
            json.dump(raw_summaries, f, indent=4)

        print(f"Processed: {filename}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--Raw_stage_folder",
        type=str,
        required=True,
        help="Directory containing raw-stage JSON files.",
    )

    parser.add_argument(
        "--output_raw_scores_dir",
        type=str,
        required=True,
        help="Directory for extracted anomaly scores.",
    )

    parser.add_argument(
        "--output_raw_summary_dir",
        type=str,
        required=True,
        help="Directory for extracted descriptions/summaries.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        args.Raw_stage_folder,
        args.output_raw_scores_dir,
        args.output_raw_summary_dir,
    )
