# ============================================================
# FILE: PartB_Practical5_NeuralStyleTransfer.py
# STANDALONE FILE — No other files needed.
# ⚠ GPU recommended but CPU works too (just slower)
#
# ── HOW TO RUN ────────────────────────────────────────────────
# Google Colab (RECOMMENDED):
#   1. Runtime → Change runtime type → T4 GPU → Save
#   2. !pip install tensorflow pillow matplotlib numpy
#   3. Paste code → Shift+Enter
#   4. Done in ~2-3 minutes on GPU
#
# Local / PyCharm (CPU):
#   pip install tensorflow pillow matplotlib numpy
#   python PartA_Practical7_NeuralStyleTransfer.py
#   (reduce epochs=3, steps_per_epoch=20 for quick test)
#
# EXPECTED OUTPUT:
#   Epoch 1/10 completed.
#   ...
#   Epoch 10/10 completed.
#   3-panel figure: content | style | stylized result
#   Strong visible style — colors, textures, brushstroke patterns
#   clearly painted over the content image structure.
#
# ── KEY DESIGN DECISION (why this works) ──────────────────────
# Images are loaded in [0.0, 1.0] float range.
# VGG19 preprocessing (vgg19.preprocess_input) is applied INSIDE the model
# by multiplying inputs by 255 first. This keeps all loss values small and
# consistent, so style_weight=1e-2 and content_weight=1e4 are correctly balanced.
# Previous attempts loaded images in [0, 255] range which inflated loss scales
# by 255^2 = 65025x, making weight tuning unpredictable.
# ============================================================

# Import TensorFlow — the deep learning framework for neural network operations
import tensorflow as tf
# Import numpy for numerical array operations
import numpy as np
# Import matplotlib for displaying images and plotting
import matplotlib.pyplot as plt
# Import VGG19 preprocessing utility from Keras
from tensorflow.keras.applications import vgg19


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

# VGG19 layers for content and style feature extraction
# Content: one deep layer captures high-level structure (objects, shapes)
content_layers = ['block5_conv2']

# Style: five layers at different depths capture multi-scale textures
# block1 = fine grain pixels/edges, block5 = high-level style patterns
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                'block4_conv1', 'block5_conv1']

num_content_layers = len(content_layers)  # 1
num_style_layers   = len(style_layers)    # 5

# Loss weights — balance between preserving content vs applying style
# These work correctly because images are in [0,1] space (not [0,255])
# style_weight=1e-2 is intentionally small because raw style loss (Gram matrix MSE)
# produces very large values; content_weight=1e4 compensates for tiny content loss values
style_weight   = 1e-2   # β — how strongly to apply the painting's style
content_weight = 1e4    # α — how strongly to preserve original content structure

# Optimization settings
EPOCHS          = 10    # Number of training epochs
STEPS_PER_EPOCH = 100   # Gradient steps per epoch → total = 1000 steps
LEARNING_RATE   = 0.02  # Adam optimizer step size


# ============================================================
# STEP 2: IMAGE LOADING
# Images are loaded as float32 in range [0.0, 1.0]
# This is CRITICAL — keeps loss scales consistent with weight values
# ============================================================
def load_and_process_image(image_path, max_dim=512):
    """
    Loads image from path or URL, resizes to max_dim (preserving aspect ratio),
    and returns as float32 tensor in [0.0, 1.0] with batch dimension added.
    [0,1] range is critical for correct loss scaling with our chosen weights.
    """
    img = tf.io.read_file(image_path)
    # Decode image to uint8 RGB tensor (H, W, 3)
    img = tf.image.decode_image(img, channels=3)
    # Convert uint8 [0,255] → float32 [0.0, 1.0]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize while keeping aspect ratio — scale based on longest dimension
    shape    = tf.cast(tf.shape(img)[:-1], tf.float32)  # (H, W) as floats
    long_dim = max(shape)                                # Longest side
    scale    = max_dim / long_dim                        # Scale factor
    new_shape = tf.cast(shape * scale, tf.int32)         # New (H, W)

    img = tf.image.resize(img, new_shape)   # Resize to new dimensions
    img = img[tf.newaxis, :]                # Add batch dim: (H,W,3) → (1,H,W,3)
    return img


# ============================================================
# STEP 3: DOWNLOAD SAMPLE IMAGES
# Using TensorFlow's hosted images for reproducibility
# Content: Yellow Labrador photo | Style: Kandinsky abstract painting
# ============================================================
print("Downloading sample images...")

# content_path = tf.keras.utils.get_file(
#     'YellowLabradorLooking_new.jpg',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
# )
content_path = tf.keras.utils.get_file(
    'lena.jpg',
    'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
)
style_path = tf.keras.utils.get_file(
    'kandinsky5.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)

content_image = load_and_process_image(content_path)  # (1, H, W, 3) in [0,1]
style_image   = load_and_process_image(style_path)    # (1, H, W, 3) in [0,1]

print(f"Content image shape: {content_image.shape}")
print(f"Style image shape:   {style_image.shape}")


# ============================================================
# STEP 4: BUILD VGG19 FEATURE EXTRACTOR
# VGG19 is pre-trained on ImageNet — its filters capture rich visual features.
# We freeze its weights and only use it as a fixed feature extractor.
# The model outputs feature maps from all content + style layers simultaneously.
# ============================================================
def get_vgg_model(style_layers, content_layers):
    """
    Builds a feature extraction model from VGG19.
    Input:  raw image tensor (after preprocessing inside StyleContentModel)
    Output: list of feature maps from [style_layers..., content_layers...]
    VGG19 weights are frozen — we never update them during NST.
    """
    # Load VGG19 without classification head (include_top=False)
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False  # Freeze all VGG19 weights permanently

    # Collect output tensors from specified layer names
    style_outputs   = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]

    # Build multi-output model: input = image, outputs = feature maps
    model_outputs = style_outputs + content_outputs  # style first, then content
    return tf.keras.Model([vgg.input], model_outputs)


# ============================================================
# STEP 5: GRAM MATRIX — Style Representation
# Gram matrix captures TEXTURE/STYLE by computing channel correlations.
# It loses spatial information (where things are) and retains only
# statistical texture information (which features co-occur together).
# Formula: G_cd = Σ_ij F_ijc * F_ijd  (sum over spatial positions)
# ============================================================
def gram_matrix(input_tensor):
    """
    Computes Gram matrix for a feature map tensor.
    input_tensor shape: (batch, height, width, channels)
    Uses einsum to compute F^T * F efficiently across spatial dimensions.
    Normalizes by number of spatial locations (H*W) to keep values stable.
    Returns: (batch, channels, channels) correlation matrix.
    """
    # einsum 'bijc,bijd->bcd': for each pair of channels c,d,
    # multiply and sum over all spatial positions i,j
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    input_shape   = tf.shape(input_tensor)
    # Number of spatial locations = height × width
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    # Normalize to prevent values from exploding with large feature maps
    return result / num_locations


# ============================================================
# STEP 6: StyleContentModel — Combined Feature Extractor
# Wraps VGG19 to extract both content and style features in one call.
# Handles VGG19 preprocessing internally (multiplies [0,1] → [0,255]
# then applies ImageNet mean subtraction via vgg19.preprocess_input).
# ============================================================
class StyleContentModel(tf.keras.models.Model):
    """
    Custom Keras model that:
    1. Accepts float32 image in [0,1] range
    2. Preprocesses it for VGG19 (scale to [0,255], subtract ImageNet mean)
    3. Extracts feature maps from content and style layers
    4. Applies Gram matrix to style features
    5. Returns dict: {'content': {...}, 'style': {...}}
    """
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg              = get_vgg_model(style_layers, content_layers)
        self.style_layers     = style_layers
        self.content_layers   = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable    = False

    def call(self, inputs):
        # Scale from [0,1] back to [0,255] for VGG19 preprocessing
        inputs = inputs * 255.0
        # Apply VGG19-specific preprocessing: subtract ImageNet RGB mean,
        # convert RGB→BGR — this is what VGG19 was trained with
        preprocessed_input = vgg19.preprocess_input(inputs)

        # Forward pass through VGG19 feature extractor
        outputs = self.vgg(preprocessed_input)

        # Split outputs: first num_style_layers are style, rest are content
        style_outputs   = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        # Apply Gram matrix to each style layer output
        # This converts spatial feature maps into texture descriptors
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Package into named dicts for easy access in loss functions
        content_dict = {name: value
                        for name, value in zip(self.content_layers, content_outputs)}
        style_dict   = {name: value
                        for name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# ============================================================
# STEP 7: EXTRACT FIXED TARGETS
# Run content and style images through the model ONCE to get target features.
# These stay fixed throughout optimization — we want the generated image
# to match these targets.
# ============================================================
print("\nBuilding VGG19 feature extractor...")
extractor = StyleContentModel(style_layers, content_layers)
print("VGG19 ready. Extracting target features...")

# Fixed targets — never change during optimization
style_targets   = extractor(style_image)['style']     # Gram matrices from style image
content_targets = extractor(content_image)['content'] # Feature maps from content image

# Initialize generated image as a copy of content image (tf.Variable so it's trainable)
# The optimizer will update this variable's pixel values each step
generated_image = tf.Variable(content_image)
print("Targets extracted. Generated image initialized from content image.")


# ============================================================
# STEP 8: OPTIMIZER SETUP
# Adam optimizer updates the PIXEL VALUES of generated_image each step.
# Note: we are NOT updating any neural network weights — VGG19 stays frozen.
# beta_1=0.99 and epsilon=1e-1 are tuned for image optimization (not typical NN training)
# ============================================================
optimizer = tf.optimizers.Adam(
    learning_rate=LEARNING_RATE, beta_1=0.99, epsilon=1e-1)


# ============================================================
# STEP 9: TRAINING STEP (Single Gradient Descent Update)
# @tf.function compiles this into a TensorFlow graph for ~3x speedup
# ============================================================
@tf.function()  # Compile as TF graph for faster execution
def train_step(image):
    """
    Performs one gradient descent step on the generated image.
    1. Forward pass: extract features from generated image
    2. Compute style loss (MSE between Gram matrices)
    3. Compute content loss (MSE between feature maps)
    4. Backpropagate: compute gradient of total loss w.r.t. image pixels
    5. Update image pixels using Adam optimizer
    6. Clip pixels to valid [0, 1] range
    NOTE: Gradients flow to IMAGE PIXELS, not to VGG19 weights.
    """
    with tf.GradientTape() as tape:
        # Forward pass through feature extractor
        outputs = extractor(image)

        # ── Style Loss ────────────────────────────────────────────
        # MSE between Gram matrices of generated image and style image
        # Summed across all 5 style layers, then scaled by style_weight
        # Divided by num_style_layers to average (not sum) across layers
        style_loss = tf.add_n([
            tf.reduce_mean((outputs['style'][name] - style_targets[name])**2)
            for name in outputs['style'].keys()
        ])
        style_loss *= style_weight / num_style_layers

        # ── Content Loss ──────────────────────────────────────────
        # MSE between feature maps of generated image and content image
        # Only one content layer (block5_conv2), scaled by content_weight
        content_loss = tf.add_n([
            tf.reduce_mean((outputs['content'][name] - content_targets[name])**2)
            for name in outputs['content'].keys()
        ])
        content_loss *= content_weight / num_content_layers

        # ── Total Loss ────────────────────────────────────────────
        # Weighted sum: higher style_weight → more style, less content
        loss = style_loss + content_loss

    # Compute gradient of total loss with respect to the generated image pixels
    # (NOT w.r.t. VGG19 weights — those are frozen)
    grad = tape.gradient(loss, image)

    # Apply gradient: move image pixels in direction that reduces loss
    optimizer.apply_gradients([(grad, image)])

    # Clip pixel values to valid [0.0, 1.0] float range after each update
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


# ============================================================
# STEP 10: OPTIMIZATION LOOP
# Runs for epochs × steps_per_epoch total gradient steps.
# More steps = better quality but takes longer.
# The generated image gradually transforms from content → stylized content.
# ============================================================
print(f"\nStarting optimization: {EPOCHS} epochs × {STEPS_PER_EPOCH} steps = "
      f"{EPOCHS * STEPS_PER_EPOCH} total gradient steps")
print(f"Style weight: {style_weight} | Content weight: {content_weight}\n")

import time
start = time.time()

for n in range(EPOCHS):
    for m in range(STEPS_PER_EPOCH):
        train_step(generated_image)
    print(f"Epoch {n+1}/{EPOCHS} completed. "
          f"Time elapsed: {time.time()-start:.1f}s")

print(f"\nOptimization complete in {time.time()-start:.1f} seconds.")


# ============================================================
# STEP 11: DISPLAY RESULTS
# Show content, style, and generated image side by side
# ============================================================
plt.figure(figsize=(15, 5))

# Content image
plt.subplot(1, 3, 1)
plt.title("Content Image", fontsize=13)
plt.imshow(tf.squeeze(content_image))   # Remove batch dim for display
plt.axis('off')

# Style image
plt.subplot(1, 3, 2)
plt.title("Style Image", fontsize=13)
plt.imshow(tf.squeeze(style_image))
plt.axis('off')

# Generated (stylized) image
plt.subplot(1, 3, 3)
plt.title("Generated Artistic Image", fontsize=13, color='green')
plt.imshow(tf.squeeze(generated_image.numpy()))  # .numpy() to convert tf.Variable
plt.axis('off')

plt.suptitle("Neural Style Transfer Results", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('neural_style_transfer.png', dpi=150, bbox_inches='tight')
plt.show()
print("Result saved as 'neural_style_transfer.png'")
print("Conclusion: Neural Style Transfer Successfully Implemented.")


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Load content image (photo) and style image (painting) in [0,1] float range
# 2. Build VGG19 feature extractor — outputs from 5 style + 1 content layer
# 3. Extract FIXED targets:
#    - Style targets: Gram matrices from style image (texture descriptors)
#    - Content targets: feature maps from content image (structure descriptors)
# 4. Initialize generated image = copy of content image (as tf.Variable)
# 5. OPTIMIZATION LOOP (epochs × steps):
#    a. Forward pass: feed generated image through VGG19 → extract features
#    b. Style loss: MSE(gram(generated), gram(style)) for each style layer
#    c. Content loss: MSE(features(generated), features(content))
#    d. Total loss = β * style_loss + α * content_loss
#    e. Backpropagate: compute ∂loss/∂pixels (NOT ∂loss/∂weights)
#    f. Adam update: shift pixel values to reduce total loss
#    g. Clip pixels to [0, 1]
# 6. After 1000 steps: generated image has content structure + style textures
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Neural Style Transfer (NST)
# Original paper: Gatys et al. (2015) "A Neural Algorithm of Artistic Style"
# VGG19: Deep CNN with 19 layers (16 conv + 3 FC), trained on ImageNet.
# Key insight: CNN layers encode different image properties —
#   Lower layers: Low-level features (edges, colors, textures) = STYLE
#   Higher layers: High-level features (objects, structure) = CONTENT
# Gram Matrix: Captures style by measuring channel-to-channel correlations.
# The optimization does NOT update VGG19 weights — it updates IMAGE PIXELS.
# Applications: artistic filters, game asset stylization, fashion design,
#               media production, video style transfer.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is Neural Style Transfer (NST)?
# A1. NST is a deep learning technique that generates a new image combining:
#     - Content from a "content image" (usually a photograph)
#     - Artistic style from a "style image" (usually a painting)
#     Uses a pre-trained CNN (VGG19) to extract content and style features,
#     then optimizes a new image to minimize both content and style losses.
#
# Q2. What is the role of content and style images in NST?
# A2. Content Image: Provides structural content — objects, shapes, layout.
#     The generated image should look like the content image in terms of "what" is in it.
#     Style Image: Provides artistic style — colors, textures, brush strokes, patterns.
#     The generated image should "look like" it was painted in the style's manner.
#     The algorithm finds a generated image that satisfies BOTH simultaneously.
#
# Q3. How are CNNs used to extract content and style features in NST?
# A3. VGG19's intermediate layers act as feature extractors (not classifiers):
#     Content: Higher-level layer (block5_conv2) captures semantic content —
#     similar activations = similar objects/structure, regardless of pixel values.
#     Style: Multiple layers (block1–block5) capture textures at different scales.
#     Gram matrix of each layer captures texture statistics independent of position.
#
# Q4. Explain content loss and style loss functions in NST.
# A4. Content Loss: MSE between content layer features of generated and content image.
#     Lcontent = Σ(F_generated - F_content)² / N
#     Penalizes generated image for having different structure from content.
#     Style Loss: MSE between Gram matrices of style layers.
#     Lstyle = Σ_l (G_generated_l - G_style_l)² / N
#     Penalizes generated image for having different textures from style image.
#
# Q5. What is a Gram Matrix? Why is it used for style representation?
# A5. Gram Matrix G_cd = Σ_ij F_ijc * F_ijd — measures correlation between channels c,d.
#     Result: (channels × channels) matrix of co-occurrence statistics.
#     Why for style: captures which features tend to appear together (texture),
#     but discards spatial information (where features are located).
#     Position-invariant → captures "style" (what textures are present)
#     without encoding "content" (where things are located).
#
# Q6. List and explain the key steps of NST.
# A6. 1. Load content and style images (float [0,1])
#     2. Build VGG19 feature extractor (frozen weights)
#     3. Extract fixed style targets (Gram matrices) and content targets (features)
#     4. Initialize generated image = copy of content image (as tf.Variable)
#     5. Optimization loop:
#        - Forward pass through VGG19
#        - Compute style loss (Gram MSE) + content loss (feature MSE)
#        - Backpropagate to image pixels
#        - Update pixels with Adam optimizer
#        - Clip to [0,1]
#     6. Display final artistic image
#
# Q7. Why is VGG19 used in Neural Style Transfer?
# A7. VGG19 is pre-trained on ImageNet — its filters capture rich visual features.
#     Simple uniform architecture (3×3 conv blocks) makes layer extraction easy.
#     Pre-trained weights provide excellent hierarchical feature representations.
#     Lower layers (block1) capture fine textures; higher layers (block5) capture
#     semantic content. This separation is ideal for NST's two objectives.
#
# Q8. What is Total Variation (TV) loss in NST?
# A8. TV loss penalizes large differences between neighboring pixels.
#     LTV = Σ((x[i,j+1]-x[i,j])² + (x[i+1,j]-x[i,j])²)
#     Reduces noise and artifacts in the generated image — acts as a
#     spatial regularizer to encourage smoothness. Weighted by γ in:
#     Ltotal = α*Lcontent + β*Lstyle + γ*LTV
#
# Q9. In NST, what is being optimized — the model or the image?
# A9. The IMAGE PIXELS are optimized, NOT the VGG19 model weights.
#     VGG19 weights are FROZEN throughout the entire optimization.
#     generated_image is a tf.Variable — its pixel values are updated each step.
#     Gradients are computed w.r.t. image pixels via GradientTape.
#     This is the opposite of normal deep learning (where we optimize weights).
#
# Q10. How can NST be used in distributed computing?
# A10. NST is computationally intensive — distributed approaches include:
#      (i) GPU acceleration: single GPU processes large images much faster
#      (ii) Multi-GPU: split batch of images across GPUs in parallel
#      (iii) TF Distributed Strategy: distribute computation across machines
#      (iv) Cloud platforms: Google Colab (free GPU), AWS EC2, Azure ML
#      (v) Feed-forward networks: train a style network once, then apply instantly
#      For video NST (millions of frames), distributed computing is essential.
#
# Q11. Why are multiple style layers used instead of one?
# A11. Different layers capture different texture scales simultaneously.
#      block1: pixel-level edges/colors; block5: large-scale style patterns.
#      Combining them produces richer, multi-scale style transfer.
#
# Q12. What happens if content weight >> style weight?
# A12. Style gradient becomes negligible → image never gets stylized.
#      The generated image stays almost identical to the content image.
#      The correct balance depends on the image scale:
#      For [0,1] images: content_weight=1e4, style_weight=1e-2 works well.
#      For [0,255] images: completely different magnitudes are needed.
#
# Q13. Why initialize generated image from content (not random noise)?
# A13. Starting from content preserves structure from step 1, requiring
#      fewer steps to converge. Random noise initialization can also work
#      but produces more abstract/distorted results and needs more iterations.
#
# Q14. Why does NST need many iterations?
# A14. Each step only moves pixels a tiny amount (LR=0.02) to avoid
#      overshooting. The competing content and style constraints require
#      many small adjustments to reach a balanced result.
#
# Q15. How can NST be accelerated for real-time applications?
# A15. Use feed-forward style transfer networks (Johnson et al. 2016) —
#      train a network once to perform one specific style, then apply in
#      one forward pass (milliseconds). Trade-off: one network per style.
#      Arbitrary style networks (e.g., Magenta on TF Hub) generalize to
#      any style in a single pass using adaptive instance normalization.
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
#
# NST OBJECTIVE:
#   Generate image G that preserves content of C and style of S.
#   Total Loss = α*ContentLoss(G,C) + β*StyleLoss(G,S)
#   Optionally add: + γ*TVLoss(G)  for smoothness regularization.
#
# CONTENT REPRESENTATION:
#   Deep feature maps from higher VGG layers (block5_conv2) capture
#   semantic structure — objects, shapes, spatial layout.
#   Two images with same structure but different colors produce similar
#   activations at this layer → good for preserving "what" is in the image.
#
# STYLE REPRESENTATION:
#   Gram matrices of multiple layers (block1–block5) capture texture correlations.
#   Each Gram matrix G[l] = F[l]^T * F[l] where F[l] is the reshaped feature map.
#   Measures which features co-activate across spatial positions.
#   Multiple layers → multi-scale texture information (fine to coarse).
#
# WHY OPTIMIZE PIXELS (NOT MODEL WEIGHTS):
#   VGG19 is a frozen feature extractor — its job is just to compute losses.
#   The variable we actually want to change is the generated image G itself.
#   So we treat pixel values as the learnable parameters and backprop into them.
#   This is sometimes called "inversion" or "feature inversion."
#
# WHY IMAGE RANGE [0,1] MATTERS:
#   Raw style loss values scale as O(pixel_value^4) due to Gram matrix computation
#   (feature maps ∝ pixel values, Gram ∝ features^2, loss ∝ Gram^2).
#   [0,255] images → losses ~65025x larger than [0,1] images.
#   style_weight=1e-2 and content_weight=1e4 are calibrated for [0,1] range.
#   CRITICAL: VGG19 preprocessing (mean subtraction) is still applied internally
#   by multiplying [0,1] inputs by 255 before calling vgg19.preprocess_input.
#
# PRACTICAL TUNING TIPS:
#   - Increase content_weight → more recognizable original scene, less style
#   - Increase style_weight  → stronger painting-like effect, may lose structure
#   - Add TV loss (γ~30)    → smoother image, fewer noise artifacts
#   - More steps/epochs     → better quality but more compute time
#   - Lower LR (0.01)       → finer updates, more stable but slower convergence
#   - Higher LR (0.05)      → faster convergence but may overshoot
#   - Start from noise      → more abstract, fully stylized result
#   - Start from content    → preserves structure, faster convergence
#
# LIMITATIONS OF ITERATIVE NST:
#   - Slow — needs hundreds to thousands of gradient steps per image
#   - Cannot be applied in real-time (video, live camera)
#   - Results sensitive to hyperparameter tuning
#   - May distort fine facial/structural details at high style weights
#   - One optimization run per image (no reuse across images)
#
# FASTER ALTERNATIVES:
#   - Johnson et al. 2016: Feed-forward style network — train once per style,
#     then apply to any content image in one forward pass (~milliseconds).
#     Trade-off: need one separate network for each style.
#   - Arbitrary style networks (AdaIN): One network for any style.
#     Uses Adaptive Instance Normalization to transfer style statistics directly.
#     Google's Magenta model on TF Hub uses this approach.
#
# REAL APPLICATIONS:
#   - Artistic photo filters (Prisma app uses NST-inspired techniques)
#   - Game asset stylization (convert realistic textures to cartoon/painterly)
#   - Fashion design (apply fabric patterns to clothing shapes)
#   - Media production (consistent visual style across shots)
#   - Video style transfer (per-frame or temporally consistent methods)
#   - Interior design visualization (apply decor styles to room photos)
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
#
# 1) WHAT NST OPTIMIZES
#    - NOT model weights (VGG19 is completely frozen throughout)
#    - Pixel values of generated image G (stored as tf.Variable)
#    - GradientTape records operations on G, computes ∂Loss/∂G
#    - Adam optimizer applies those gradients to update G's pixels
#    - After each update, pixels clipped to [0,1] to stay valid
#
# 2) THREE LOSS COMPONENTS
#    Content Loss:
#      - Layer: block5_conv2 (deep, abstract, semantic)
#      - Formula: mean((F_generated - F_content)^2)
#      - Purpose: keep generated image "about the same thing" as content
#    Style Loss:
#      - Layers: block1–block5 (shallow to deep = fine to coarse textures)
#      - Formula: mean((Gram_generated_l - Gram_style_l)^2) averaged over layers
#      - Purpose: make generated image "look painted like" the style image
#    TV Loss (optional but recommended):
#      - Formula: sum of squared differences between neighboring pixels
#      - Purpose: suppresses noise and checkerboard artifacts
#      - Not included in this implementation (add if output looks noisy)
#
# 3) GRAM MATRIX IN DETAIL
#    - Input: feature map F of shape (1, H, W, C)
#    - Reshape: F → (H*W, C) — flatten spatial dimensions
#    - Compute: G = F^T * F → shape (C, C)
#    - Each entry G[c,d] = dot product of channel c and channel d across all pixels
#    - Interpretation: G[c,d] is large when features c and d co-occur spatially
#    - Normalization: divide by H*W*C to keep values scale-independent
#    - Why position-invariant: we summed over all spatial positions (i,j)
#      so the final matrix has no memory of WHERE features appeared
#
# 4) VGG19 ARCHITECTURE USED IN NST
#    Block 1: 2 conv layers (64 filters, 3×3) → captures edges, colors
#    Block 2: 2 conv layers (128 filters, 3×3) → captures simple textures
#    Block 3: 4 conv layers (256 filters, 3×3) → captures complex textures
#    Block 4: 4 conv layers (512 filters, 3×3) → captures object parts
#    Block 5: 4 conv layers (512 filters, 3×3) → captures object-level semantics
#    We use block1_conv1 through block5_conv1 for style (all scales)
#    We use block5_conv2 for content (deepest semantic layer)
#
# 5) STYLECONTENT MODEL DESIGN
#    - Wraps VGG19 as a frozen sub-model
#    - Accepts [0,1] float images as input
#    - Internally scales to [0,255] and applies VGG19 preprocessing
#    - Returns dict: {'style': {layer: gram_matrix, ...},
#                     'content': {layer: feature_map, ...}}
#    - Clean separation of preprocessing from loss computation
#    - @tf.function on train_step compiles to TF graph for ~3x speedup
#
# 6) WHY epochs × steps_per_epoch STRUCTURE
#    - Allows progress reporting after each epoch
#    - Inner loop (steps_per_epoch) runs without Python overhead per step
#    - @tf.function on train_step means the inner loop is nearly pure C++/CUDA
#    - Total steps = epochs(10) × steps_per_epoch(100) = 1000 gradient updates
#    - More steps = better convergence = stronger, cleaner style transfer
#
# 7) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "This NST implementation uses a frozen VGG19 as a fixed feature extractor.
#     Content features are extracted from block5_conv2; style features are Gram
#     matrices from block1 through block5. The generated image is initialized as
#     a copy of the content image (tf.Variable in [0,1] range) and iteratively
#     updated using Adam optimizer to minimize the weighted sum of content loss
#     (feature map MSE) and style loss (Gram matrix MSE). VGG19 weights never
#     change — only the pixel values of the generated image are optimized."
# ============================================================

# ============================================================
# ADDITIONAL HIGH-VALUE VIVA Q&A:
#
# Q16. What is the difference between iterative NST and feed-forward NST?
# A16. Iterative NST (Gatys 2015): optimizes one image at a time over many steps.
#      Slow (~minutes per image) but flexible — works with any style/content pair.
#      Feed-forward NST (Johnson 2016): trains a separate network for each style.
#      After training (~hours), applies style to any content in one pass (~ms).
#      Trade-off: iterative is flexible but slow; feed-forward is fast but inflexible.
#
# Q17. What role does ImageNet pre-training play in NST?
# A17. VGG19 trained on ImageNet has learned to detect edges, textures, patterns,
#      and objects across 1000 categories. These learned filters are general enough
#      to serve as a universal feature extractor for style and content.
#      Without pre-training, random VGG weights would extract meaningless features
#      and NST would produce random noise regardless of weights/steps.
#
# Q18. Can NST be applied to video? What are the challenges?
# A18. Yes, but naively applying per-frame NST produces flickering (temporal
#      inconsistency) because each frame is optimized independently.
#      Solutions: (i) initialize each frame from the previous stylized frame,
#      (ii) add temporal consistency loss penalizing differences between frames,
#      (iii) use feed-forward networks with temporal smoothing.
#      Distributed GPU compute is essential for real-time video NST.
#
# Q19. What happens to the generated image at step 0 vs step 1000?
# A19. Step 0: generated image = exact copy of content image. Content loss = 0.
#      Style loss is large (Gram matrices of content ≠ Gram matrices of style).
#      Steps 1-1000: optimizer pushes pixels toward style image's texture statistics
#      while the content loss term resists large structural changes.
#      Step 1000: generated image has content structure but style colors/textures.
#      Content loss rises slightly (pixels shifted away from original content features).
#      Style loss decreases significantly (Gram matrices converge toward style targets).
#
# Q20. Why does NST use MSE (Mean Squared Error) for both losses?
# A20. MSE is differentiable everywhere — allows clean gradient computation via
#      backpropagation. It penalizes large deviations more than small ones
#      (quadratic penalty), pushing the optimizer strongly when far from target
#      and gently when close. Alternatives like MAE (L1) are less commonly used
#      because they have undefined gradients at zero and produce sparser solutions.
# ============================================================