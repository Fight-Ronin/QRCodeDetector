# QR Code Detection and Decoding

This document outlines an optimized approach for detecting and decoding QR codes in images using classical image processing techniques. Our method is designed to accurately locate QR codes within a frame and interpret their contents, offering robust performance even in challenging conditions.

## Overview

The QR code detection process comprises several key steps, from initial image preprocessing to the final extraction and decoding of QR code data. Here's a concise overview of the method:

1. **Preprocessing**: Apply image preprocessing techniques to enhance QR code visibility. This includes noise reduction and contrast enhancement.
2. **Finder Pattern Detection**: Locate the unique finder patterns of QR codes using size ratio methods and connected component analysis.
3. **Grouping Finder Patterns**: Identify groups of finder patterns to pinpoint QR code locations.
4. **QR Code Bounding Box Identification**: Extrapolate from the finder patterns to determine the bounding box of the entire QR code.

## Detailed Process

### Dependencies
All required Library are in the environment.yml folder. Run main.py to execute the QRCode recognizer.

### 1. Image Preprocessing
- Utilize Sauvola binarization for effective contrast enhancement and noise reduction, preparing the image for further analysis.

### 2. Finder Pattern Detection
- Implement a pixel-by-pixel analysis to detect the characteristic 1:1:3:1:1 size ratio of QR code finder patterns.
- Employ connected component analysis to find clusters of pixels that likely constitute finder patterns.

### 3. Grouping Finder Patterns
- Construct right-angled triangles with detected finder patterns and validate the presence of a quiet zone around the QR code, adhering to the QR code specifications.

### 4. QR Code Bounding Box Identification
- Calculate extensions from the finder patterns to approximate the QR code's corners and sides, facilitating the identification of the entire QR code's bounding box.

## Decoding and Displaying QR Code Data

Upon detecting QR codes, the system decodes their contents, handling various data types such as URLs, text, Wi-Fi credentials, and contact information. The decoded data is displayed on the screen as long as the QR code remains in the frame, ensuring real-time feedback and interaction.

## Performance and Adaptations

The algorithm achieves efficient detection at approximately 20 frames per second, with jitter mitigation through the use of a Kalman filter. This filter smooths the bounding box's movement across frames, providing stable and reliable QR code tracking.

## Advanced Considerations

For specific scenarios involving known QR codes with perspective transformations, additional techniques such as feature matching and homography computation can be employed. However, due to the uniform appearance of QR codes, direct decoding and string comparison are recommended for reliability.

## Conclusion

This method offers a comprehensive and effective solution for QR code detection and decoding, combining classical image processing techniques with modern filtering and data interpretation strategies. It is adaptable to a range of conditions and capable of handling various types of encoded data.

## Functionalities
1. Able to recognize the data type in QRcode, meanwhile outputting the content on the detection screen
2. Able to open websites directly if detected inside a QRcode (for safty purpose only opened https:// format websites)
