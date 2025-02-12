import torch.nn as nn  # Importing the neural network module from PyTorch
import torch  # Importing PyTorch

# Define the Encoder class, inheriting from nn.Module
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()  # Call the parent class (nn.Module) constructor

        # Define the sequential model for the encoder
        self.model = nn.Sequential(
            # Input state (3x256x256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.InstanceNorm2d(16, affine=True),  # Instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation

            # State (16x128x128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.InstanceNorm2d(32, affine=True),  # Instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation

            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.InstanceNorm2d(64, affine=True),  # Instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation

            # State (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.InstanceNorm2d(128, affine=True),  # Instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation

            # State (128x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.InstanceNorm2d(256, affine=True),  # Instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation

            # State (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.InstanceNorm2d(512, affine=True),  # Instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation

            # State (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0)  # Final convolutional layer
            # Output state (1024x1x1)
        )

        # Define the last layer to reduce the feature map to the latent dimension
        self.last_layer = nn.Sequential(
            nn.Linear(1024, latent_dim),  # Linear layer to reduce to latent dimension
            nn.Linear(latent_dim, latent_dim)  # Another linear layer for further processing
        )

    # Define the forward pass
    def forward(self, img):
        features = self.model(img)  # Pass the input through the sequential model
        features = features.view(img.shape[0], -1)  # Flatten the feature map
        features = self.last_layer(features)  # Pass through the last layer
        features = features.view(features.shape[0], -1, 1, 1)  # Reshape to (batch_size, latent_dim, 1, 1)
        return features  # Return the encoded features

# Define the Decoder class, inheriting from nn.Module
class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()  # Call the parent class (nn.Module) constructor
        self.img_shape = img_shape  # Store the image shape

        # Define the sequential model for the decoder
        self.model = nn.Sequential(
            # Input state (latent_dim x 1 x 1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),  # Transposed convolutional layer
            nn.BatchNorm2d(num_features=1024),  # Batch normalization
            nn.LeakyReLU(True),  # Leaky ReLU activation

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),  # Transposed convolutional layer
            nn.BatchNorm2d(num_features=512),  # Batch normalization
            nn.LeakyReLU(True),  # Leaky ReLU activation

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),  # Transposed convolutional layer
            nn.BatchNorm2d(num_features=256),  # Batch normalization
            nn.LeakyReLU(True),  # Leaky ReLU activation

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # Transposed convolutional layer
            nn.BatchNorm2d(num_features=128),  # Batch normalization
            nn.LeakyReLU(True),  # Leaky ReLU activation

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),  # Transposed convolutional layer
            nn.BatchNorm2d(num_features=64),  # Batch normalization
            nn.LeakyReLU(True),  # Leaky ReLU activation

            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  # Transposed convolutional layer
            nn.BatchNorm2d(num_features=32),  # Batch normalization
            nn.LeakyReLU(True),  # Leaky ReLU activation

            # State (32x128x128)
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),  # Final transposed convolutional layer
            nn.Tanh()  # Tanh activation to scale output to [-1, 1]
        )

    # Define the forward pass
    def forward(self, input):
        img = self.model(input)  # Pass the input through the sequential model
        return img.view(img.shape[0], *self.img_shape)  # Reshape to (batch_size, channels, height, width)

# Create instances of the Encoder and Decoder
encoder = Encoder(latent_dim=100)  # Instantiate the Encoder with a latent dimension of 100
decoder = Decoder(img_shape=(3, 256, 256), latent_dim=100)  # Instantiate the Decoder with image shape (3, 256, 256) and latent dimension of 100

# Print the architectures
print(encoder)  # Print the Encoder architecture
print(decoder)  # Print the Decoder architecture