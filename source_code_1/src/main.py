#!/usr/bin/env python3
"""
Quantum Circuit Extraction Pipeline - Main Entry Point

This script orchestrates the entire pipeline:
1. Load configuration and paper list
2. For each paper: download PDF, extract images, detect circuits
3. For detected circuits: extract gates, classify problem, compile metadata
4. Save outputs: circuit images, JSON dataset, CSV tracking
5. Support resume via checkpointing

Usage:
    python src/main.py
    python src/main.py --max-circuits 100
    python src/main.py --paper-list input/test_papers.txt
"""

import sys
import os
import time
import logging
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from utils import ConfigLoader, ensureDirectory, CheckpointData
from input_handler import InputHandler
from download_manager import DownloadManager
from pdf_image_extractor import PdfImageExtractor
from detectors.detection_orchestrator import DetectionOrchestrator
from metadata_compiler import MetadataCompiler
from checkpoint_manager import CheckpointManager
from output_generator import OutputGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProjectNLP.Main")


def printBanner():
    """Print pipeline startup banner."""
    print("=" * 70)
    print("QUANTUM CIRCUIT EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def runPipeline(
    paperListPath: str = "input/paper_list_1.txt",
    maxCircuits: int = 250,
    configPath: str = "config/config.yaml"
):
    """
    Run the complete quantum circuit extraction pipeline.
    
    Args:
        paperListPath: Path to file containing arXiv IDs
        maxCircuits: Stop after collecting this many circuits
        configPath: Path to configuration file
    """
    startTime = time.time()
    
    # Ensure working directory is project root so relative paths resolve
    projectRoot = Path(__file__).parent.parent
    os.chdir(projectRoot)
    
    # Print startup info
    printBanner()
    print(f"Paper list: {paperListPath}")
    print(f"Max circuits: {maxCircuits}")
    print("=" * 70)
    
    # =========================================================================
    # PHASE 1: INITIALIZATION
    # =========================================================================
    print("\n[PHASE 1] INITIALIZATION")
    print("-" * 40)
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader()
    config.load(configPath)
    logger.info("Pipeline started")
    
    # Get config values
    maxCircuitCount = config.get("pipeline.maxCircuitCount", maxCircuits)
    if maxCircuits != 250:  # Override if user specified
        maxCircuitCount = maxCircuits
    cleanTempAfterPaper = config.get("pipeline.cleanTempAfterPaper", True)
    tempPdfsDir = config.get("paths.temp.pdfsDir", "temp/pdfs")
    
    # Create module instances
    print("Initializing modules...")
    
    # Module 1: Input Handler - reads paper list
    inputHandler = InputHandler(paperListPath)
    
    # Module 2: Download Manager - downloads PDFs from arXiv
    downloadManager = DownloadManager(config)
    
    # Module 3: PDF Image Extractor - extracts images from PDFs
    pdfImageExtractor = PdfImageExtractor(config)
    
    # Module 4: Detection Orchestrator - detects circuits, extracts gates, classifies problems
    print("  - Loading DINOv2 detector (this may take a moment)...")
    detectionOrchestrator = DetectionOrchestrator(config)
    
    # Module 5: Metadata Compiler - compiles metadata for detected circuits
    metadataCompiler = MetadataCompiler(config)
    
    # Module 6: Checkpoint Manager - saves/restores progress
    checkpointManager = CheckpointManager(config)
    
    # Module 7: Output Generator - saves images, JSON, CSV
    outputGenerator = OutputGenerator(config=config)
    
    # Load paper list
    print("Loading paper list...")
    loadResult = inputHandler.load()
    if not loadResult.loadSuccess:
        print(f"ERROR: Failed to load paper list: {loadResult.errors}")
        return
    
    paperIds = loadResult.paperIds
    totalPapers = loadResult.totalCount
    print(f"  Loaded {totalPapers} papers")
    
    # Load existing outputs for resume
    outputGenerator.loadExisting()
    
    # Get resume point from checkpoint
    startIndex, currentCircuitCount = checkpointManager.getResumePoint()
    
    if startIndex > 0:
        print(f"  Resuming from paper index {startIndex}")
        print(f"  Circuits already found: {currentCircuitCount}")
        outputGenerator.setImageCounter(currentCircuitCount)
    else:
        print("  Starting fresh run")
    
    # Statistics tracking
    papersProcessed = 0
    papersSkippedNotQuantPh = 0
    papersFailed = 0
    paperIndex = startIndex - 1  # safe default if loop never runs
    lastPaperId = ""
    
    print("\n" + "=" * 70)
    print("[PHASE 2] MAIN PROCESSING LOOP")
    print("=" * 70)
    
    # =========================================================================
    # PHASE 2: MAIN PROCESSING LOOP
    # =========================================================================
    try:
        # Create progress bar
        pbar = tqdm(
            range(startIndex, totalPapers),
            desc="Processing papers",
            initial=startIndex,
            total=totalPapers
        )
        
        for paperIndex in pbar:
            arxivId = paperIds[paperIndex]
            lastPaperId = arxivId
            
            # Check stopping condition
            if currentCircuitCount >= maxCircuitCount:
                logger.info(f"Reached max circuits ({maxCircuitCount}). Stopping.")
                print(f"\n✓ Reached target: {maxCircuitCount} circuits collected!")
                break
            
            # Update progress bar description
            pbar.set_description(f"Paper {arxivId} | Circuits: {currentCircuitCount}/{maxCircuitCount}")
            
            circuitsInPaper = 0
            
            try:
                # -----------------------------------------------------------------
                # Step 1: Download PDF
                # -----------------------------------------------------------------
                downloadResult = downloadManager.download(arxivId)
                
                if not downloadResult.downloadSuccess:
                    if not downloadResult.isQuantPh:
                        # Paper is not in quant-ph category
                        papersSkippedNotQuantPh += 1
                        outputGenerator.updateCsv(arxivId, 0)
                        continue
                    else:
                        # Download failed
                        papersFailed += 1
                        logger.warning(f"Failed to download {arxivId}: {downloadResult.error}")
                        outputGenerator.updateCsv(arxivId, 0)
                        continue
                
                pdfPath = downloadResult.pdfPath
                
                # -----------------------------------------------------------------
                # Step 2: Extract images from PDF
                # -----------------------------------------------------------------
                extractionResult = pdfImageExtractor.extract(
                    pdfPath=pdfPath,
                    arxivId=arxivId,
                    paperTitle=downloadResult.paperTitle,
                    paperAbstract=downloadResult.paperAbstract
                )
                
                if not extractionResult.success:
                    papersFailed += 1
                    logger.warning(f"Image extraction failed for {arxivId}")
                    outputGenerator.updateCsv(arxivId, 0)
                    continue
                
                pdfData = extractionResult.pdfData
                images = extractionResult.images
                
                # -----------------------------------------------------------------
                # Step 3: Process each extracted image
                # -----------------------------------------------------------------
                for imageInfo in images:
                    # Check stopping condition again
                    if currentCircuitCount >= maxCircuitCount:
                        break
                    
                    # Detect if image is a quantum circuit
                    detectionResult = detectionOrchestrator.detect(
                        imagePath=imageInfo.tempPath,
                        captionText=imageInfo.captionText or "",
                        surroundingText=imageInfo.surroundingText or "",
                        pdfData=pdfData,
                        figureNumber=imageInfo.captionFigureNum,
                        imageInfo=imageInfo  # NEW: Pass complete ImageInfo
                    )
                    
                    if detectionResult.isQuantumCircuit:
                        # Circuit detected!
                        
                        # Compile metadata
                        metadata = metadataCompiler.compile(
                            detectionResult=detectionResult,
                            imageInfo=imageInfo,
                            pdfData=pdfData
                        )
                        
                        # Save circuit image and metadata
                        outputGenerator.saveCircuit(
                            imagePath=imageInfo.tempPath,
                            metadata=metadata
                        )
                        
                        currentCircuitCount += 1
                        circuitsInPaper += 1
                        
                        logger.info(f"Circuit {currentCircuitCount} found in {arxivId}")
                
                # Update paper count in CSV
                outputGenerator.updateCsv(arxivId, circuitsInPaper)
                papersProcessed += 1
                lastPaperId = arxivId
                
                # Save checkpoint after each paper
                checkpointData = CheckpointData(
                    lastPaperIndex=paperIndex,
                    lastPaperId=arxivId,
                    totalCircuitsFound=currentCircuitCount,
                    papersProcessed=papersProcessed,
                    papersSkippedNotQuantPh=papersSkippedNotQuantPh,
                    papersFailed=papersFailed,
                    timestamp=datetime.now().isoformat()
                )
                checkpointManager.save(checkpointData)
                
                # Clean up temp PDF if configured
                if cleanTempAfterPaper and pdfPath and os.path.exists(pdfPath):
                    os.remove(pdfPath)
                if cleanTempAfterPaper:
                    tempImagesDir = Path(tempPdfsDir).parent / "images" / arxivId
                    shutil.rmtree(tempImagesDir, ignore_errors=True)
                    
            except Exception as e:
                logger.error(f"Error processing {arxivId}: {str(e)}")
                papersFailed += 1
                outputGenerator.updateCsv(arxivId, 0)
                continue
        
        pbar.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Saving progress...")
        safePaperIndex = max(paperIndex, -1)
        safePaperId = lastPaperId
        checkpointData = CheckpointData(
            lastPaperIndex=safePaperIndex,
            lastPaperId=safePaperId,
            totalCircuitsFound=currentCircuitCount,
            papersProcessed=papersProcessed,
            papersSkippedNotQuantPh=papersSkippedNotQuantPh,
            papersFailed=papersFailed,
            timestamp=datetime.now().isoformat()
        )
        checkpointManager.save(checkpointData)
        outputGenerator.saveAll()
        print("Progress saved. You can resume by running the pipeline again.")
        return
    
    # =========================================================================
    # PHASE 3: FINALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("[PHASE 3] FINALIZATION")
    print("=" * 70)
    
    # Mark remaining papers as not processed (empty in CSV)
    if totalPapers > 0 and paperIndex + 1 < totalPapers:
        remainingPapers = paperIds[paperIndex + 1:]
        outputGenerator.markRemainingBlank(remainingPapers)
        print(f"  Marked {len(remainingPapers)} remaining papers as not processed")
    
    # Save all outputs (atomic writes)
    print("Saving outputs...")
    outputGenerator.saveAll()
    
    # Clear checkpoint (pipeline completed successfully)
    checkpointManager.clear()
    
    # Calculate elapsed time
    elapsedTime = time.time() - startTime
    elapsedMinutes = int(elapsedTime // 60)
    elapsedSeconds = int(elapsedTime % 60)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total circuits collected: {currentCircuitCount}")
    print(f"Papers processed: {papersProcessed}")
    print(f"Papers skipped (not quant-ph): {papersSkippedNotQuantPh}")
    print(f"Papers failed: {papersFailed}")
    print(f"Time elapsed: {elapsedMinutes}m {elapsedSeconds}s")
    
    # Print caption filter statistics
    detectionOrchestrator.caption_filter.print_statistics()
    
    print("\nOutput files:")
    print(f"  - Images: output/images_1/")
    print(f"  - Dataset: output/dataset_1.json")
    print(f"  - Tracking: output/paper_list_counts_1.csv")
    print("=" * 70)
    
    logger.info(f"Pipeline completed: {currentCircuitCount} circuits in {elapsedMinutes}m {elapsedSeconds}s")


def main():
    """Parse command line arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Quantum Circuit Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/main.py
    python src/main.py --max-circuits 100
    python src/main.py --paper-list input/test_papers.txt
        """
    )
    
    parser.add_argument(
        "--paper-list",
        default="input/paper_list_1.txt",
        help="Path to paper list file (default: input/paper_list_1.txt)"
    )
    
    parser.add_argument(
        "--max-circuits",
        type=int,
        default=250,
        help="Maximum circuits to collect (default: 250)"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)"
    )
    
    args = parser.parse_args()
    
    runPipeline(
        paperListPath=args.paper_list,
        maxCircuits=args.max_circuits,
        configPath=args.config
    )


if __name__ == "__main__":
    main()
