#!/usr/bin/env python3
"""
Command-line interface for microCT analysis.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from . import MicroCTAnalyzer, visualize_results

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated microCT analysis from BMP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
            Examples:
            
            # Basic analysis
            python cli.py /path/to/bmp/files --output results.json

            # Analysis with custom parameters
            python cli.py /path/to/bmp/files --segmentation-method watershed --min-area 200 --output results.json

            # Analysis with visualization
            python cli.py /path/to/bmp/files --output results.json --visualize --save-plots plots/

            # Analysis of specific slice range
            python cli.py /path/to/bmp/files --slice-range 10 50 --output results.json
        """
    )

    # Required arguments
    parser.add_argument(
        'input_directory',
        help='Directory containing BMP files'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Output file path for results (JSON, CSV, or Excel)',
        required=True
    )

    # Analysis parameters
    parser.add_argument(
        '--segmentation-method',
        choices=['otsu', 'local', 'watershed', 'manual'],
        default='otsu',
        help='Segmentation method (default: otsu)'
    )

    parser.add_argument(
        '--min-area',
        type=int,
        default=100,
        help='Minimum area for regions (default: 100)'
    )

    parser.add_argument(
        '--slice-range',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help='Range of slices to analyze (start end)'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize volume (default: True)'
    )

    parser.add_argument(
        '--normalization-method',
        choices=['minmax', 'zscore', 'histogram'],
        default='minmax',
        help='Normalization method (default: minmax)'
    )

    # Visualization options
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations'
    )

    parser.add_argument(
        '--save-plots',
        help='Directory to save plot images'
    )

    # Additional options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--summary-report',
        action='store_true',
        help='Print summary report to console'
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Validate input directory
    if not os.path.isdir(args.input_directory):
        logger.error(f"Input directory does not exist: {args.input_directory}")
        sys.exit(1)

    # Create analyzer
    analyzer = MicroCTAnalyzer(
        segmentation_method=args.segmentation_method,
        min_area=args.min_area,
        normalize=args.normalize,
        normalization_method=args.normalization_method
    )

    try:
        # Run analysis
        logger.info(f"Starting analysis of {args.input_directory}")

        results = analyzer.run_complete_analysis(
            directory=args.input_directory,
            slice_range=tuple(args.slice_range) if args.slice_range else None
        )

        # Save results
        output_format = Path(args.output).suffix.lower().lstrip('.')
        if output_format == 'json':
            format_type = 'json'
        elif output_format == 'csv':
            format_type = 'csv'
        elif output_format in ['xlsx', 'xls']:
            format_type = 'excel'
        else:
            format_type = 'json'

        analyzer.save_results(args.output, format=format_type)

        # Print summary report
        if args.summary_report:
            print("\n" + analyzer.get_summary_report())

        # Create visualizations
        if args.visualize:
            logger.info("Creating visualizations")
            visualize_results(results, save_dir=args.save_plots)

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
