import argparse
import sys
import logging
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ before config is parsed

# transformers imports TransformGetItemToIndex from a path that torch 2.10 reorganized.
# Expose it at the old path so transformers loads correctly.
import torch._dynamo._trace_wrapped_higher_order_op as _thop
if not hasattr(_thop, "TransformGetItemToIndex"):
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex as _T
    _thop.TransformGetItemToIndex = _T

from cmpq.core.config import CompressorConfig
from cmpq.core.artifact_manager import ArtifactManager
from cmpq.pipeline import CMPQPipeline

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    parser = argparse.ArgumentParser(description="CMP-Q post-training weight quantization")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--set", metavar="section.key=value", nargs="*", dest="overrides",
                        help="Dot-notation overrides, e.g. --set quantization.bits=3 experiment.name=my_run")
    parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment name")
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("Main")

    # 1. Load Config
    try:
        config = CompressorConfig.load_from_yaml(args.config, overrides=args.overrides)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if args.experiment_name:
        config.experiment.name = args.experiment_name

    # 2. Setup Artifacts
    artifacts = ArtifactManager(config.experiment.output_dir, config.experiment.name)
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Output Dir: {artifacts.root}")

    # 3. Run CMP-Q Stage
    logger.info("=== Running CMP-Q Pipeline ===")
    stage_instance = CMPQPipeline(config, artifacts)
    
    try:
        success = stage_instance.run()
        if not success:
            logger.error("CMP-Q Pipeline failed.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"CMP-Q Pipeline crashed: {e}")
        sys.exit(1)

    logger.info("CMP-Q Pipeline completed successfully.")

if __name__ == "__main__":
    main()
