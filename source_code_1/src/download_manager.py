"""
Module 2: Download Manager
Validate paper category is "quant-ph" and download PDF if valid.
"""

import logging
import arxiv
from pathlib import Path
from typing import Optional

from utils import DownloadResult, ensureDirectory, ConfigLoader

logger = logging.getLogger("ProjectNLP.DownloadManager")


class DownloadManager:
    """Handle arXiv paper validation and PDF download."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize DownloadManager.
        
        Args:
            config: ConfigLoader instance
        """
        self.outputDir = config.get("paths.temp.pdfsDir", "temp/pdfs")
        self.validCategory = config.get("arxiv.validCategory", "quant-ph")
        
        ensureDirectory(self.outputDir)
    
    def download(self, arxivId: str) -> DownloadResult:
        """
        Query arXiv for paper metadata and download PDF if quant-ph.
        
        Args:
            arxivId: arXiv paper ID (e.g., "2501.18147")
            
        Returns:
            DownloadResult with metadata and download status
        """
        logger.info(f"Processing paper: {arxivId}")
        
        # Check if PDF already exists locally (but still call API for metadata)
        filename = f"{arxivId.replace('/', '_')}.pdf"
        pdfPath = Path(self.outputDir) / filename
        pdf_exists_locally = pdfPath.exists()
        
        # Initialize result
        result = DownloadResult(
            arxivId=arxivId,
            isQuantPh=False,
            categories=[],
            downloadSuccess=False,
            pdfPath=None,
            paperTitle="",
            paperAbstract="",
            authors=[],
            error=None
        )
        
        # =====================================================================
        # ALWAYS CALL API to get title and abstract (needed for classification)
        # =====================================================================
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxivId])
            paper = next(client.results(search), None)
            
            if paper is None:
                result.error = f"Paper not found: {arxivId}"
                logger.warning(result.error)
                return result
            
            result.paperTitle = paper.title
            result.paperAbstract = paper.summary
            result.authors = [author.name for author in paper.authors]
            result.categories = list(paper.categories)
            
            logger.debug(f"Paper title: {result.paperTitle}")
            logger.debug(f"Categories: {result.categories}")
            
            result.isQuantPh = self.validCategory in result.categories
            
            if not result.isQuantPh:
                logger.info(f"Paper {arxivId} is not quant-ph. Categories: {result.categories}")
                return result
            
            # Download PDF only if not exists locally
            if pdf_exists_locally:
                logger.info(f"✓ PDF exists locally: {arxivId}")
                result.pdfPath = str(pdfPath)
                result.downloadSuccess = True
            else:
                logger.info(f"Downloading PDF: {arxivId}")
                downloaded_path = self._downloadPdf(paper, arxivId)
                
                if downloaded_path:
                    result.pdfPath = downloaded_path
                    result.downloadSuccess = True
                    logger.info(f"Successfully downloaded: {downloaded_path}")
                else:
                    result.error = "PDF download failed"
                    logger.error(f"Failed to download PDF for {arxivId}")
                
        except StopIteration:
            result.error = f"Paper not found: {arxivId}"
            logger.warning(result.error)
        except Exception as e:
            result.error = f"Error processing paper {arxivId}: {str(e)}"
            logger.error(result.error)
        
        return result
    
    def _downloadPdf(self, paper: arxiv.Result, arxivId: str) -> Optional[str]:
        """Download PDF for a paper."""
        try:
            filename = f"{arxivId.replace('/', '_')}.pdf"
            pdfPath = Path(self.outputDir) / filename
            
            paper.download_pdf(dirpath=self.outputDir, filename=filename)
            
            if pdfPath.exists():
                return str(pdfPath)
            else:
                return None
                
        except Exception as e:
            logger.error(f"PDF download error: {str(e)}")
            return None
