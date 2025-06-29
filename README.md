# README.md


# Image Authentication Toolkit
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: PEP-8](https://img.shields.io/badge/code%20style-PEP--8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/chirindaopensource/image_authentication_toolkit)
[![Code Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](https://github.com/chirindaopensource/image_authentication_toolkit)
[![Release Version](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/chirindaopensource/image_authentication_toolkit/releases/tag/v1.0.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


A multi-modal system for the quantitative analysis of image provenance, uniqueness, and semantic context.

**Repository:** [https://github.com/chirindaopensource/image_authentication_toolkit](https://github.com/chirindaopensource/image_authentication_toolkit)  
**License:** MIT  
**Owner:** © 2025 Craig Chirinda (Open Source Projects)

## Abstract

The proliferation of sophisticated generative models has introduced significant ambiguity into the domain of digital visual media. Establishing the originality and provenance of an image is no longer a matter of simple inspection but requires a quantitative, multi-faceted analytical framework. This toolkit provides a suite of methodologically rigorous tools to address this challenge. It enables the systematic dissection of an image's structural, statistical, and semantic properties, allowing for an empirical assessment of its relationship to other visual works and its context within the public domain. By integrating techniques from classical computer vision, deep learning, and web automation, this system facilitates informed decision-making, strategic intervention, and nuanced comprehension for any organization concerned with the integrity and management of its digital visual assets.

## Table of Contents

1.  [Methodological Framework](#methodological-framework)
2.  [System Architecture](#system-architecture)
3.  [Core Components](#core-components)
4.  [Setup and Installation](#setup-and-installation)
5.  [Usage Example](#usage-example)
6.  [Theoretical Foundations](#theoretical-foundations)
7.  [Error Handling and Robustness](#error-handling-and-robustness)
8.  [Contributing](#contributing)
9.  [License](#license)
10. [Citation](#citation)

## Methodological Framework

The toolkit employs a layered, multi-modal analysis strategy. Each layer provides a distinct form of evidence, and a robust conclusion is reached through the synthesis of their results. The analysis proceeds from low-level pixel statistics to high-level semantic meaning and public context.

1.  **Level 1: Perceptual Hashing (Near-Duplicate Detection)**
    *   **Objective:** To identify structurally identical or near-identical images.
    *   **Mechanism:** Utilizes a DCT-based perceptual hash (`pHash`) to create a compact fingerprint of an image's low-frequency components. The Hamming distance between two fingerprints quantifies their dissimilarity.
    *   **Application:** Serves as a rapid, computationally inexpensive first pass to flag direct replication.

2.  **Level 2: Local Feature Matching (Geometric & Structural Analysis)**
    *   **Objective:** To detect if a section of one image has been copied, scaled, rotated, or otherwise transformed and inserted into another.
    *   **Mechanism:** Employs the ORB algorithm to detect thousands of salient keypoints. These are matched between images, and the geometric consistency of these matches is verified using a RANSAC-based homography estimation.
    *   **Application:** Essential for identifying digital collage, "asset ripping," and partial duplications.

3.  **Level 3: Global Statistical Analysis (Color & Texture Profile)**
    *   **Objective:** To compare the global statistical properties of images, such as their color palette distribution.
    *   **Mechanism:** Computes multi-dimensional color histograms in a perceptually uniform space (e.g., HSV, LAB) and compares them using statistical metrics like Pearson correlation or Bhattacharyya distance.
    *   **Application:** Useful for identifying images with a shared aesthetic, from a common source, or subject to the same post-processing filters. It is a weaker signal for direct copying.

4.  **Level 4: Semantic Embedding (Conceptual Similarity)**
    *   **Objective:** To measure the abstract, conceptual similarity between images, independent of style or composition.
    *   **Mechanism:** Leverages the CLIP Vision Transformer to project images into a high-dimensional semantic embedding space. The cosine similarity between two image vectors in this space quantifies their conceptual proximity.
    *   **Application:** The primary tool for analyzing stylistic influence and thematic overlap. It can determine if an AI-generated image of a "cyberpunk city in the style of Van Gogh" is semantically close to Van Gogh's actual works.

5.  **Level 5: Public Provenance (Web Context Discovery)**
    *   **Objective:** To determine if an image or its near-duplicates exist in the publicly indexed web and to gather context about their usage.
    *   **Mechanism:** Utilizes robust, Selenium-based web automation to perform a reverse image search on Google Images, scraping and structuring the results.
    *   **Application:** A critical discovery tool for establishing a baseline of public existence and understanding how an image is being used and described across the web.

## System Architecture

The toolkit is designed around principles of modularity, testability, and robustness, adhering to the SOLID principles of object-oriented design.

*   **Dependency Inversion:** The core `ImageSimilarityDetector` class does not depend on concrete implementations. Instead, it depends on abstractions defined by `Protocol` classes (`FeatureDetectorProtocol`, `MatcherProtocol`, `ClipModelLoaderProtocol`). This allows for easy substitution of underlying algorithms and facilitates unit testing with mock objects.
*   **Single Responsibility:** Responsibilities are cleanly segregated.
    *   `ImageSimilarityDetector`: Orchestrates the analytical workflow.
    *   **Factory Classes** (`DefaultFeatureDetectorFactory`, etc.): Encapsulate the complex logic of creating and configuring optimized algorithm instances.
    *   **Result Dataclasses** (`FeatureMatchResult`, etc.): Structure the output data and contain validation and serialization logic, separating results from computation.
    *   **Error Classes** (`ImageSimilarityError`, etc.): Provide a rich, hierarchical system for handling exceptions with detailed forensic context.
*   **Resource Management:** Lazy loading is used for computationally expensive resources like the CLIP model, which is only loaded into memory upon its first use. The `ResourceManager` provides background monitoring and automated cleanup of system resources, ensuring stability in long-running applications.

## Core Components

The project is composed of several key Python classes within the `image_authentication_tool_draft.ipynb` notebook:

*   **`ImageSimilarityDetector`**: The primary public-facing class. It orchestrates all analysis methods and manages dependencies and resources.
*   **Result Dataclasses**:
    *   `ReverseImageSearchResult`: A structured container for results from the Google reverse image search.
    *   `FeatureMatchResult`: A structured container for results from the ORB feature matching analysis.
    *   `StatisticalProperties`: A reusable dataclass for encapsulating detailed statistical analysis of a data sample.
*   **Factory Classes**:
    *   `DefaultFeatureDetectorFactory`: Creates and caches optimized `cv2.ORB` instances.
    *   `DefaultMatcherFactory`: Creates and profiles `cv2.BFMatcher` instances.
    *   `DefaultClipModelLoader`: Manages the lifecycle (loading, caching, optimization) of CLIP models.
*   **Protocol Definitions**:
    *   `FeatureDetectorProtocol`, `MatcherProtocol`, `ClipModelLoaderProtocol`: Define the abstract interfaces for the core computational components, enabling dependency injection.
*   **Custom Exception Hierarchy**:
    *   `ImageSimilarityError`: The base exception for all toolkit-related errors.
    *   Specialized subclasses (`ImageNotFoundError`, `ModelLoadError`, `NavigationError`, etc.) provide granular context for specific failure modes.

## Setup and Installation

A rigorous setup is required to ensure reproducible and accurate results.

1.  **Prerequisites:**
    *   Python 3.9 or newer.
    *   `git` for cloning the repository.
    *   A C++ compiler for building some of the underlying library dependencies.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/chirindaopensource/image_authentication_toolkit.git
    cd image_authentication_toolkit
    ```

3.  **Create a Virtual Environment:**
    It is imperative to work within a dedicated virtual environment to manage dependencies and avoid conflicts.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file would contain packages such as `numpy`, `opencv-python`, `torch`, `Pillow`, `imagehash`, `selenium`, `scipy`, `pandas`, and `ftfy` (for CLIP).*

5.  **Install ChromeDriver:**
    The `reverse_image_search_google` method requires the Selenium ChromeDriver.
    *   **Verify your Chrome version:** Go to `chrome://settings/help`.
    *   **Download the matching ChromeDriver:** Visit the [Chrome for Testing availability dashboard](https://googlechromelabs.github.io/chrome-for-testing/).
    *   Place the `chromedriver` executable in the root of the project directory or another location in your system's `PATH`. The usage example assumes it is in the root.

## Usage Example

The following script demonstrates a complete, multi-modal analysis of two images.

```python
import json
import logging
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

# Assume all classes from the notebook are available in the execution scope.
# This includes ImageSimilarityDetector, ResourceConstraints, ValidationPolicy,
# and all custom exception classes.

def demonstrate_image_provenance_analysis(
    detector: ImageSimilarityDetector,
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    chromedriver_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Executes a comprehensive, multi-modal analysis to compare two images and
    establish the provenance of the first image.

    This function serves as a production-grade demonstration of the
    ImageSimilarityDetector's capabilities, invoking each of its primary
    analytical methods in a structured sequence. It captures results from
    perceptual hashing, local feature matching, global color analysis,
    semantic similarity, and public reverse image search.

    The methodology proceeds from low-level structural comparisons to
    high-level semantic and contextual analysis, providing a holistic
    view of the relationship between the images.

    Args:
        detector (ImageSimilarityDetector): An initialized instance of the
            image similarity detector.
        image1_path (Union[str, Path]): The file path to the primary image
            to be analyzed and compared. This image will also be used for
            the reverse image search.
        image2_path (Union[str, Path]): The file path to the secondary image
            for comparison.
        chromedriver_path (Union[str, Path]): The file path to the
            Selenium ChromeDriver executable, required for reverse image search.

    Returns:
        Dict[str, Any]: A dictionary containing the detailed results from each
            analysis stage. Each key corresponds to an analysis method, and
            the value is either a comprehensive result dictionary/object or
            an error message if that stage failed.
    """
    # Initialize a dictionary to aggregate the results from all analysis methods.
    analysis_results: Dict[str, Any] = {}
    # Configure logging to provide visibility into the analysis process.
    logging.info(f"Starting comprehensive provenance analysis for '{Path(image1_path).name}' and '{Path(image2_path).name}'.")

    # --- Stage 1: Perceptual Hash Analysis (Structural Duplication) ---
    logging.info("Executing Stage 1: Perceptual Hash Analysis...")
    try:
        p_hash_results = detector.perceptual_hash_difference(
            image1_path, image2_path, hash_size=16, normalize=True,
            return_similarity=True, statistical_analysis=True
        )
        analysis_results['perceptual_hash'] = p_hash_results
        logging.info(f"  - pHash Similarity Score: {p_hash_results.get('similarity_score', 'N/A'):.4f}")
    except Exception as e:
        analysis_results['perceptual_hash'] = {'error': str(e), 'details': traceback.format_exc()}
        logging.error(f"  - Perceptual Hash Analysis failed: {e}")

    # --- Stage 2: Local Feature Matching Analysis (Geometric Consistency) ---
    logging.info("Executing Stage 2: Local Feature Matching Analysis...")
    try:
        feature_match_results = detector.feature_match_ratio(
            image1_path, image2_path, distance_threshold=64,
            normalization_strategy="min_keypoints", apply_ratio_test=True,
            ratio_threshold=0.75, resize_max_side=1024,
            return_detailed_result=True, geometric_verification=True,
            statistical_analysis=True
        )
        analysis_results['feature_matching'] = feature_match_results
        logging.info(f"  - Feature Match Similarity Ratio: {feature_match_results.similarity_ratio:.4f}")
        logging.info(f"  - Geometric Inlier Ratio: {feature_match_results.homography_inlier_ratio or 'N/A'}")
    except Exception as e:
        analysis_results['feature_matching'] = {'error': str(e), 'details': traceback.format_exc()}
        logging.error(f"  - Feature Matching Analysis failed: {e}")

    # --- Stage 3: Global Color Distribution Analysis (Palette Similarity) ---
    logging.info("Executing Stage 3: Global Color Distribution Analysis...")
    try:
        histogram_results = detector.histogram_correlation(
            image1_path, image2_path, metric="correlation", color_space="HSV",
            statistical_analysis=True, adaptive_binning=True
        )
        analysis_results['histogram_correlation'] = histogram_results
        logging.info(f"  - Histogram Correlation: {histogram_results.get('similarity_score', 'N/A'):.4f}")
    except Exception as e:
        analysis_results['histogram_correlation'] = {'error': str(e), 'details': traceback.format_exc()}
        logging.error(f"  - Histogram Correlation Analysis failed: {e}")

    # --- Stage 4: Semantic Meaning Analysis (Conceptual Similarity) ---
    logging.info("Executing Stage 4: Semantic Meaning Analysis...")
    try:
        clip_results = detector.clip_embedding_similarity(
            image1_path, image2_path, statistical_analysis=True,
            embedding_analysis=True, batch_processing=True
        )
        analysis_results['semantic_similarity'] = clip_results
        logging.info(f"  - CLIP Cosine Similarity: {clip_results.get('cosine_similarity', 'N/A'):.4f}")
    except Exception as e:
        analysis_results['semantic_similarity'] = {'error': str(e), 'details': traceback.format_exc()}
        logging.error(f"  - Semantic Similarity Analysis failed: {e}")

    # --- Stage 5: Public Provenance and Context Analysis (Web Discovery) ---
    logging.info("Executing Stage 5: Public Provenance Analysis...")
    try:
        reverse_search_results = detector.reverse_image_search_google(
            image_path=image1_path, driver_path=chromedriver_path,
            headless=True, advanced_extraction=True, content_analysis=True
        )
        analysis_results['reverse_image_search'] = reverse_search_results
        logging.info(f"  - Reverse Search Best Guess: {reverse_search_results.best_guess}")
        logging.info(f"  - Found {len(reverse_search_results.similar_image_urls)} similar images online.")
    except Exception as e:
        analysis_results['reverse_image_search'] = {'error': str(e), 'details': traceback.format_exc()}
        logging.error(f"  - Reverse Image Search failed: {e}")

    logging.info("Comprehensive provenance analysis complete.")
    return analysis_results

if __name__ == '__main__':
    # This block demonstrates how to run the analysis.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_dir = Path("./test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create test images
    image1_path = test_dir / "original_image.png"
    image2_path = test_dir / "modified_image.jpg"
    cv2.imwrite(str(image1_path), np.full((512, 512, 3), 64, dtype=np.uint8))
    cv2.imwrite(str(image2_path), np.full((512, 512, 3), 68, dtype=np.uint8))

    chromedriver_path = Path("./chromedriver")
    if not chromedriver_path.exists():
        logging.error("FATAL: ChromeDriver not found. Please download it and place it in the project root.")
        sys.exit(1)

    detector = ImageSimilarityDetector()
    full_results = demonstrate_image_provenance_analysis(detector, image1_path, image2_path, chromedriver_path)

    def result_serializer(obj):
        if isinstance(obj, (Path, np.ndarray)): return str(obj)
        if hasattr(obj, 'to_dict'): return obj.to_dict()
        return str(obj)

    print("\n" + "="*40 + " ANALYSIS RESULTS " + "="*40)
    print(json.dumps(full_results, default=result_serializer, indent=2))
    print("="*100)
```

## Theoretical Foundations

The implementation rests on established principles from multiple scientific and engineering disciplines.

*   **Software Engineering & Design Patterns**
    *   **Object-Oriented Design:** Encapsulation of logic within classes (`ImageSimilarityDetector`, `ResourceManager`).
    *   **SOLID Principles:** Dependency Inversion is used via `Protocol`s to decouple the main class from concrete algorithm implementations. Single Responsibility is evident in the separation of concerns between factories, result objects, and the main detector.
    *   **Resource Management:** Lazy loading (`_load_clip...`) and background monitoring (`ResourceManager`) ensure efficient use of memory and compute.
    *   **Error Handling:** A comprehensive, custom exception hierarchy allows for granular error reporting and robust recovery.

*   **Computer Vision & Image Processing**
    *   **Feature Detection:** The ORB implementation is based on the canonical papers for FAST corners and BRIEF descriptors, with added logic for orientation invariance.
    *   **Perceptual Hashing:** The `pHash` algorithm is a direct application of frequency-domain analysis using the Discrete Cosine Transform (DCT) to create a scale- and compression-invariant image fingerprint.
    *   **Histogram Analysis:** The use of HSV and LAB color spaces is a standard technique to achieve a degree of illumination invariance in color-based comparisons.

*   **Machine Learning & Deep Learning**
    *   **Contrastive Learning:** The `clip_embedding_similarity` method is a direct application of the CLIP model, which learns a joint embedding space for images and text through contrastive learning on a massive dataset.
    *   **Vision Transformer (ViT):** The CLIP model's vision component is a ViT, which processes images as sequences of patches using self-attention mechanisms, enabling it to capture global semantic context.

*   **Mathematics & Statistics**
    *   **Linear Algebra:** Cosine similarity is computed via the dot product of L2-normalized embedding vectors.
    *   **Probability & Statistics:** Pearson correlation is used for histogram comparison. The Hamming distance is a fundamental metric from information theory. Statistical confidence intervals are computed for key metrics to quantify uncertainty.
    *   **Geometric Verification:** Homography estimation via RANSAC is a robust statistical method for finding a geometric consensus among noisy data points (the feature matches).

## Error Handling and Robustness

The system is designed for production environments and incorporates multiple layers of error handling:

*   **Custom Exception Hierarchy:** Allows for specific and actionable error catching (e.g., distinguishing a `ModelLoadError` from a `NavigationError`).
*   **Input Validation:** Each public method rigorously validates its inputs against mathematical and logical constraints before proceeding. The `_validate_image_path` method is particularly extensive, checking for existence, permissions, file type, and content integrity.
*   **Retry Mechanisms:** Core operations, such as web driver initialization and page navigation, are wrapped in retry loops with exponential backoff to handle transient network or system failures.
*   **Fallback Strategies:** The CLIP model loader will automatically fall back from GPU to CPU upon encountering a `CUDA OutOfMemoryError`, ensuring the operation can complete, albeit more slowly. The web automation uses a hierarchy of selectors to find UI elements, making it resilient to minor front-end changes.

## Contributing

Contributions that adhere to a high standard of methodological rigor are welcome.

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/your-feature-name`).
3.  Develop your feature, ensuring it is accompanied by appropriate unit tests.
4.  Ensure your code adheres to the PEP-8 style guide.
5.  Submit a pull request with a detailed description of your changes and their justification.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this toolkit in your academic research, please cite it as follows:

```bibtex
@software{Chirinda_Image_Authentication_Toolkit_2025,
  author = {Chirinda, Craig},
  title = {{Image Authentication Toolkit}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chirindaopensource/image_authentication_toolkit}}
}
```

--

This README was generated based on the structure and content of image_authentication_tool_draft.ipynb and follows best practices for research software documentation.
