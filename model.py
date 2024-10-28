import torch
import torch.nn as nn
from models.baseline import BaselineCaptioninngModel
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

class ModelHandler:

    def __init__(self, model_type, device, model_weights=None, vocab_required=True, tokenizer=None, **kwargs):
        """
        Initialize the ModelHandler.

        Args:
            model_type (str): The type of model ('baseline', 'CLIP', 'BLIP').
            device (torch.device): Device where the model will be loaded.
            model_weights (str, optional): Path to the custom model weights.
            vocab_required (bool): Whether the model requires a vocabulary.
            tokenizer (AutoTokenizer, optional): Pretrained tokenizer if using CLIP or BLIP.
            kwargs: Additional parameters for model initialization (e.g., vocab_size, embed_size).
        """
        self.model_type = model_type
        self.device = device
        self.vocab_required = vocab_required
        self.tokenizer = tokenizer
        self.model = None
        self.processor = None  # Only used for some models like CLIP or BLIP
        self.load_model(model_weights, **kwargs)

    def load_model(self, model_weights=None, **kwargs):
        """Loads the appropriate model based on the model_type and loads weights if provided."""
        
        if self.model_type == "baseline":
            # Load the Baseline CNN-RNN Image Captioning Model
            embed_size = kwargs.get("embed_size", 256)
            hidden_size = kwargs.get("hidden_size", 512)
            vocab_size = kwargs.get("vocab_size", 5000)
            num_layers = kwargs.get("num_layers", 1)
            
            self.model = BaselineCaptioninngModel(embed_size, hidden_size, vocab_size, num_layers)
            self.model = self.model.to(self.device)

            # Load custom weights if provided
            if model_weights:
                print(f"Loading custom weights from {model_weights}...")
                self.model.load_state_dict(torch.load(model_weights, map_location=self.device))

        elif self.model_type == "CLIP":
            # Load CLIP model from Hugging Face
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

            if model_weights:
                print(f"Loading custom weights from {model_weights}...")
                self.model.load_state_dict(torch.load(model_weights, map_location=self.device))

        elif self.model_type == "BLIP":
            # Load BLIP model for conditional generation
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

            if model_weights:
                print(f"Loading custom weights from {model_weights}...")
                self.model.load_state_dict(torch.load(model_weights, map_location=self.device))

        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")

    def forward_train(self, images, captions, lengths, vocab_required=True, tokenizer=None):
        """
        Perform the forward pass for training.

        Args:
            images (torch.Tensor): The batch of images.
            captions (torch.Tensor): The batch of tokenized captions.
            lengths (list): List of caption lengths for packing padded sequences.
            vocab_required (bool): Whether the model requires a vocabulary.
            tokenizer (AutoTokenizer, optional): Tokenizer for pretrained models.

        Returns:
            torch.Tensor: Output from the forward pass.
        """
        if self.model_type == "baseline":
            # Custom CNN-RNN baseline model training forward pass
            features = self.model.encoder(images)
            outputs = self.model.decoder(features, captions, lengths)
            return outputs

        elif self.model_type == "CLIP":
            # Fine-tune CLIP model
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(input_ids=captions)  # Or relevant method for text-image training
            return outputs

        elif self.model_type == "BLIP":
            # Fine-tune BLIP model
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(input_ids=captions, pixel_values=inputs['pixel_values'])
            return outputs.logits  # Return the logits for the training objective

        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")

    def generate_caption(self, images, vocab=None, lengths=None):
        """
        Generate captions for the given images.
        Args:
            images (torch.Tensor): The batch of images.
            vocab (Vocabulary): The vocabulary object (only for vocab-based models like baseline).
            lengths (list): List of caption lengths.

        Returns:
            list: Generated captions as a list of token indices.
        """
        if self.model_type == "baseline":
            # Generate captions using the Baseline model (CNN-RNN)
            with torch.no_grad():
                features = self.model.encoder(images.to(self.device))
                captions = self.model.decoder.generate(features, vocab, lengths)
            return captions

        elif self.model_type == "CLIP":
            # CLIP does not generate captions but we can return image-text embeddings for evaluation
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            return outputs

        elif self.model_type == "BLIP":
            # Generate captions using BLIP
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            return captions

        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")
