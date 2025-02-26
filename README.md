# Image Compression via Latent Code Optimization (Experimental)

This project is an experimental approach to image compression using a simple neural network in C++. Instead of storing full bitmaps, we “compress” each image into a small latent code that a shared decoder then uses to reconstruct the image. This is more of a playful test to see if we can make an interesting trade-off between compression and quality rather than a final solution.

---

## What’s It All About?

We start by generating three synthetic images:
- **Radial Gradient:** A smooth gradient radiating from the center.
- **Sine Wave Pattern:** A wavy, periodic pattern.
- **Checkerboard:** A classic black-and-white grid.

Each image gets its own latent vector (a kind of compressed version of the image). A single shared decoder—a simple linear transformation followed by a sigmoid function—reconstructs the images from their latent codes.

The training process adjusts both the decoder and each latent code to minimize the difference between the original and reconstructed images. OpenMP helps speed up computations.

---

## How It Works

### **Image Generation and Saving**
   - The program generates three synthetic images.
   - Each image is saved as a PGM file (a simple grayscale format).

### **Latent Codes and Decoder**
   - Each image gets a unique latent code (a vector of fixed size).
   - The decoder applies a transformation plus a sigmoid activation to produce an image.

### **Training Process**
   - The decoder reconstructs an image from each latent code.
   - The error (mean squared difference) between the reconstruction and the original is computed.
   - The gradients update both the decoder and each latent code.
   - Progress (loss and elapsed time) is printed every 10 epochs.

### **Reconstruction**
   - Each optimized latent code is used to reconstruct an image.
   - The final images are saved for comparison.

---

## Getting Started

### Compilation

To compile, run:

```bash
g++ -O2 -fopenmp -std=c++17 main.cpp -o neural_compression
```

### Running the Program

After compilation, execute:

```bash
./neural_compression
```

This will:
- Generate and save three synthetic images.
- Train for **1000 epochs**.
- Save the reconstructed images as:
  - `final_decoded_0.pgm`
  - `final_decoded_1.pgm`
  - `final_decoded_2.pgm`

---

## Comparing Results

Once training is complete, the generated images and their reconstructed versions can be compared side by side. This helps in understanding where the compression method succeeds and where it fails.

### Common Artifacts in Reconstruction

#### **Blurriness**
   - The reconstructed images tend to lose fine details.
   - This is because the decoder is limited in capacity, meaning high-frequency details are harder to recover.

#### **Distortions in Patterns**
   - The checkerboard pattern might show irregularities.
   - Sine waves may have slight phase shifts or unexpected oscillations.
   - This is due to the decoder learning an average behavior rather than pixel-perfect reconstructions.

####  **Loss of Contrast**
   - The radial gradient might not reach perfect black or white.
   - This happens because the sigmoid activation limits the dynamic range.

To improve these, we’d need a more expressive decoder, better optimization strategies, or a more structured latent space.



---

## Final Thoughts

This isn’t meant to be a cutting-edge compression technique, but rather a fun experiment to see how well a neural network can store information in a small latent space. The results show both promise and limitations, making it a great starting point for further experiments!

Feel free to tweak the settings, try different latent sizes, or add more complex images.

---

## Contribute

Want to improve this experiment? Feel free to:
- Fork and experiment with modifications.
- Open an issue if you have suggestions.
- Share your results with different images!

---

## License

This project is open-source under the MIT License. Modify and use freely!

