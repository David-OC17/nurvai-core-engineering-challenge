from pathlib import Path

import typer

from nurvai_pipeline.config import PipelineConfig
from nurvai_pipeline.pipeline import run_pipeline
from nurvai_pipeline.qa_render import render_qa_video

app = typer.Typer(add_completion=False)


@app.command()
def run(
    video: Path = typer.Option(..., exists=True, help="Path to video.mp4"),
    imu: Path = typer.Option(..., exists=True, help="Path to imu.csv"),
    vts: Path = typer.Option(..., exists=True, help="Path to vts.csv"),
    output_dir: Path = typer.Option(Path("out"), help="Directory for enriched.jsonl / action_chunks.jsonl"),
):
    """Run the full alignment + hand-tracking + segmentation pipeline."""
    run_pipeline(video, imu, vts, output_dir, PipelineConfig())


@app.command()
def qa(
    video: Path = typer.Option(..., exists=True, help="Path to the original video.mp4"),
    jsonl: Path = typer.Option(..., exists=True, help="Path to enriched.jsonl produced by `run`"),
    output: Path = typer.Option(Path("out/output_qa.mp4"), help="Path to write the QA verification video"),
):
    """Render a verification video overlaying keypoints and telemetry on the original video."""
    output.parent.mkdir(parents=True, exist_ok=True)
    render_qa_video(video, jsonl, output)
    typer.echo(f"Wrote QA video to {output}")


if __name__ == "__main__":
    app()
