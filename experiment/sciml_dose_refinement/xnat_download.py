#!/usr/bin/env python3
"""
Script to download all data from XNAT project
"""

import os
import sys
import xnat
from pathlib import Path
import logging
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
XNAT_SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
XNAT_ALIAS = "5ce23897-3467-4932-a29e-e256d77aea35"
XNAT_SECRET = "x4G94myEK6umDnJcEEU6jUuFGbVReywm6HeDwEddEhZYvcpZsb6d2KKJzHOygAZa"
PROJECT_ID = "Lu117 recon"
DOWNLOAD_DIR = "/DATA"


def create_download_directory(base_dir):
    """Create the download directory if it doesn't exist"""
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download directory: {path.absolute()}")
    return path


def download_project_data(session, project_id, download_dir):
    """Download all data from a project"""
    try:
        project = None
        for pid, p in session.projects.items():
            name = p.name.lower()
            if ("lu117" in name or "lu177" in name) and "recon" in name:
                project = p
                project_id = pid
                break
                
        if project is None:
            raise KeyError(project_id)
            
        logger.info(f"Found project: {project.name} (ID: {project_id})")

        # Get all subjects in the project
        subjects = list(project.subjects.values())
        logger.info(f"Found {len(subjects)} subjects in project")

        total_files = 0
        total_size = 0
        successful = 0
        failed = 0
        downloaded_patients = []

        for subject_idx, subject in enumerate(subjects, 1):
            subject_label = subject.label
            logger.info(
                f"\n[{subject_idx}/{len(subjects)}] Processing subject: {subject_label}"
            )

            # Create subject directory
            subject_dir = download_dir / subject_label
            if not subject_dir.exists():
                logger.warning(f"  Skipping subject {subject_label} because its folder does not exist in {download_dir}")
                continue

            # We need to check both subject-level and experiment-level resources
            all_resources = list(subject.resources.values())

            # Get all experiments for this subject
            experiments = list(subject.experiments.values())
            for exp in experiments:
                all_resources.extend(list(exp.resources.values()))
            
            logger.info(f"      Found {len(all_resources)} total resources")

            for resource_idx, resource in enumerate(all_resources, 1):
                resource_label = resource.label
                logger.info(
                    f"        [{resource_idx}/{len(all_resources)}] Resource: {resource_label}"
                )

                # Get all files in this resource
                try:
                    files = list(resource.files.values())
                    logger.info(f"          Found {len(files)} files")

                    for file_idx, file in enumerate(files, 1):
                        file_name = file.name
                        if file_name.lower() != "dosemap.nii.gz":
                            continue
                        try:
                            file_size = int(file.size) if hasattr(file, "size") and file.size else 0
                        except ValueError:
                            file_size = 0

                        try:
                            dosemap_dir = subject_dir / "DOSEMAP"
                            dosemap_dir.mkdir(parents=True, exist_ok=True)
                            file_path = dosemap_dir / file_name

                            # Download the file
                            logger.info(
                                f"            [{file_idx}/{len(files)}] Downloading: {file_name} ({file_size / (1024*1024):.2f} MB)"
                            )
                            file.download(str(file_path))

                            total_files += 1
                            total_size += file_size
                            logger.info(
                                f"              ✓ Downloaded to: {file_path}"
                            )
                            successful += 1
                            
                            if subject_label not in downloaded_patients:
                                downloaded_patients.append(subject_label)

                        except Exception as e:
                            logger.error(
                                f"              ✗ Failed to download {file_name}: {e}"
                            )
                            failed += 1

                except Exception as e:
                    logger.error(
                        f"        Error accessing files in resource {resource_label}: {e}"
                    )

            logger.info(f"  Completed subject: {subject_label}")

        logger.info(f"\n{'='*70}")
        logger.info("Download Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"✓ Successful files: {successful}")
        logger.info(f"✗ Failed files: {failed}")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total size: {total_size / (1024*1024*1024):.2f} GB")
        logger.info(f"Download directory: {download_dir.absolute()}")
        logger.info(f"{'='*70}")
        
        # Write CSV report
        csv_path = download_dir / "downloaded_dosemaps.csv"
        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Subject_Label"])
                for patient in downloaded_patients:
                    writer.writerow([patient])
            logger.info(f"CSV Report saved to: {csv_path} (Total Patients with Dosemap: {len(downloaded_patients)})")
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")

    except KeyError:
        logger.error(f"Project '{project_id}' not found on XNAT server")
        logger.info("\nAvailable projects:")
        for proj_id in session.projects.keys():
            logger.info(f"  - {proj_id}")
        return False
    except Exception as e:
        logger.error(f"Error downloading project data: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False

    return True


def main():
    """Main function to download XNAT project data"""
    import argparse

    parser = argparse.ArgumentParser(description="Download data from XNAT project")
    parser.add_argument("--server", type=str, default=None, help="XNAT server URL")
    parser.add_argument("--project", type=str, default=None, help="XNAT project ID")
    parser.add_argument(
        "--output",
        type=str,
        default=DOWNLOAD_DIR,
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    # Override defaults if provided
    server = args.server if args.server else XNAT_SERVER
    project = args.project if args.project else PROJECT_ID
    output_dir = args.output

    logger.info("=" * 70)
    logger.info("XNAT Download Script")
    logger.info("=" * 70)
    logger.info(f"XNAT Server: {server}")
    logger.info(f"Project: {project}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("\nDRY RUN MODE - No files will be downloaded")
        logger.info("Run without --dry-run to perform actual download")
        return

    # Create download directory
    download_path = create_download_directory(output_dir)

    # Connect to XNAT
    logger.info(f"\nConnecting to XNAT server: {server}")
    try:
        with xnat.connect(
            server=server, user=XNAT_ALIAS, password=XNAT_SECRET
        ) as session:
            logger.info("✓ Successfully connected to XNAT")

            # Download project data
            download_project_data(session, project, download_path)

    except Exception as e:
        logger.error(f"Failed to connect to XNAT: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
