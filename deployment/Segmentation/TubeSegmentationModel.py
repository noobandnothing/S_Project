
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

class TubeSegmentationModel:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Load deployment package
        package = torch.load(model_path, map_location=self.device)
        
        # Recreate model
        self.model = smp.Unet(
            encoder_name=package['model_config']['encoder_name'],
            encoder_weights=None,  # Don't load pretrained for deployment
            in_channels=package['model_config']['in_channels'],
            classes=package['model_config']['classes'],
            activation=package['model_config']['activation']
        )
        
        # Load trained weights
        self.model.load_state_dict(package['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=package['preprocessing']['normalize_mean'],
                std=package['preprocessing']['normalize_std']
            )
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Training info: {package['training_info']}")
    
    def predict_and_crop_exact_shape(self, image_path, output_dir=None, threshold=0.5, 
                                   padding=10, transparent_background=True, 
                                   edge_smoothing=True, feather_edges=2):
        """
        Predict mask and create exact shape cropped image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs (if None, saves in same dir as input)
            threshold: Confidence threshold for binary mask
            padding: Extra pixels around the detected object
            transparent_background: If True, creates PNG with transparent background
            edge_smoothing: Apply Gaussian blur to smooth edges
            feather_edges: Pixels to feather the edges (0 = no feathering)
        
        Returns:
            dict with paths to saved files and metadata
        """
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size
        
        input_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize mask back to original size using OpenCV for better quality
        mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply edge smoothing if requested
        if edge_smoothing:
            mask_smooth = cv2.GaussianBlur(mask_resized, (5, 5), 1.0)
            binary_mask = (mask_smooth > threshold).astype(np.float32)
            
            # Apply feathering for smooth edges
            if feather_edges > 0:
                binary_uint8 = (binary_mask * 255).astype(np.uint8)
                dist_transform = cv2.distanceTransform(binary_uint8, cv2.DIST_L2, 5)
                feather_mask = np.minimum(dist_transform / feather_edges, 1.0)
                feather_mask = np.where(binary_mask > 0, feather_mask, 0)
                alpha_mask = (feather_mask * 255).astype(np.uint8)
            else:
                alpha_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = (mask_resized > threshold).astype(np.float32)
            alpha_mask = (binary_mask * 255).astype(np.uint8)
        
        # Setup output paths
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        # Save mask
        mask_pil = Image.fromarray(alpha_mask, 'L')
        mask_pil.save(mask_path)
        
        # Check if object detected
        if alpha_mask.max() == 0:
            return {
                'success': False,
                'mask_path': mask_path,
                'cropped_path': None,
                'error': 'No object detected in image',
                'confidence_score': float(pred_mask.max()),
                'mask_area_ratio': 0.0
            }
        
        # Find bounding box of the mask
        coords = np.where(alpha_mask > 10)  # Small threshold to include feathered edges
        top, left = coords[0].min(), coords[1].min()
        bottom, right = coords[0].max(), coords[1].max()
        
        # Add padding to bounding box
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(original_size[0], right + padding)
        bottom = min(original_size[1], bottom + padding)
        
        # Crop the original image and mask to bounding box
        original_array = np.array(original_image)
        cropped_image = original_array[top:bottom+1, left:right+1]
        cropped_alpha = alpha_mask[top:bottom+1, left:right+1]
        
        # Create different output versions
        results = {}
        
        if transparent_background:
            # Create RGBA image with transparent background
            cropped_path = os.path.join(output_dir, f"{base_name}_exact_shape.png")
            
            rgba_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = cropped_image  # RGB channels
            rgba_image[:, :, 3] = cropped_alpha   # Alpha channel from mask
            
            # Save as PNG with transparency
            rgba_pil = Image.fromarray(rgba_image, 'RGBA')
            rgba_pil.save(cropped_path)
            results['cropped_path'] = cropped_path
            
        else:
            # Create image with white background where mask is 0
            cropped_path = os.path.join(output_dir, f"{base_name}_exact_shape_white_bg.png")
            
            mask_3d = np.stack([cropped_alpha, cropped_alpha, cropped_alpha], axis=2) / 255.0
            white_background = np.ones_like(cropped_image) * 255
            final_image = (cropped_image * mask_3d + white_background * (1 - mask_3d)).astype(np.uint8)
            
            final_pil = Image.fromarray(final_image)
            final_pil.save(cropped_path)
            results['cropped_path'] = cropped_path
        
        # Always create a black background version for comparison
        black_bg_path = os.path.join(output_dir, f"{base_name}_exact_shape_black_bg.png")
        mask_3d = np.stack([cropped_alpha, cropped_alpha, cropped_alpha], axis=2) / 255.0
        black_background = np.zeros_like(cropped_image)
        black_bg_image = (cropped_image * mask_3d + black_background * (1 - mask_3d)).astype(np.uint8)
        black_bg_pil = Image.fromarray(black_bg_image)
        black_bg_pil.save(black_bg_path)
        results['black_bg_path'] = black_bg_path
        
        result = {
            'success': True,
            'mask_path': mask_path,
            'bbox': (left, top, right, bottom),
            'cropped_size': (right - left, bottom - top),
            'confidence_score': float(pred_mask.max()),
            'mask_area_ratio': float(binary_mask.sum() / binary_mask.size),
            'exact_shape': True,
            'transparent_background': transparent_background,
            'edge_smoothing': edge_smoothing,
            'feather_edges': feather_edges
        }
        result.update(results)
        
        return result
    
    def predict_and_crop_exact_shape_obj(self, image, threshold=0.5, 
                                   padding=10, transparent_background=True, 
                                   edge_smoothing=True, feather_edges=2):
        if isinstance(image, np.ndarray):
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_image = Image.fromarray(image)
            original_size = original_image.size
        
        input_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize mask back to original size using OpenCV for better quality
        mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply edge smoothing if requested
        if edge_smoothing:
            mask_smooth = cv2.GaussianBlur(mask_resized, (5, 5), 1.0)
            binary_mask = (mask_smooth > threshold).astype(np.float32)
            
            # Apply feathering for smooth edges
            if feather_edges > 0:
                binary_uint8 = (binary_mask * 255).astype(np.uint8)
                dist_transform = cv2.distanceTransform(binary_uint8, cv2.DIST_L2, 5)
                feather_mask = np.minimum(dist_transform / feather_edges, 1.0)
                feather_mask = np.where(binary_mask > 0, feather_mask, 0)
                alpha_mask = (feather_mask * 255).astype(np.uint8)
            else:
                alpha_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = (mask_resized > threshold).astype(np.float32)
            alpha_mask = (binary_mask * 255).astype(np.uint8)
        
        
        
        

        # Find bounding box of the mask
        coords = np.where(alpha_mask > 10)  # Small threshold to include feathered edges
        top, left = coords[0].min(), coords[1].min()
        bottom, right = coords[0].max(), coords[1].max()
        
        # Add padding to bounding box
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(original_size[0], right + padding)
        bottom = min(original_size[1], bottom + padding)
        
        # Crop the original image and mask to bounding box
        original_array = np.array(original_image)
        cropped_image = original_array[top:bottom+1, left:right+1]
        cropped_alpha = alpha_mask[top:bottom+1, left:right+1]
        
        # Create different output versions
        
        if transparent_background:
            # Create RGBA image with transparent background
            
            rgba_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = cropped_image  # RGB channels
            rgba_image[:, :, 3] = cropped_alpha   # Alpha channel from mask
            
            # Save as PNG with transparency
            rgba_pil = Image.fromarray(rgba_image, 'RGBA')
            
                
        
        return rgba_pil
    
    def batch_process(self, input_dir, output_dir, threshold=0.5, padding=10, 
                     transparent_background=True, edge_smoothing=True, feather_edges=2):
        """
        Process multiple images in a directory with exact shape cropping
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        results = []
        print(f"Processing {len(image_files)} images with exact shape cropping...")
        
        for filename in image_files:
            image_path = os.path.join(input_dir, filename)
            try:
                result = self.predict_and_crop_exact_shape(
                    image_path, output_dir, threshold, padding,
                    transparent_background, edge_smoothing, feather_edges
                )
                result['filename'] = filename
                results.append(result)
                
                if result['success']:
                    print(f"✅ {filename} - Exact shape extracted: {result['cropped_size']}")
                else:
                    print(f"❌ {filename} - {result['error']}")
                    
            except Exception as e:
                print(f"❌ {filename} - Error: {str(e)}")
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
        
        # Save summary
        successful = sum(1 for r in results if r.get('success', False))
        summary_path = os.path.join(output_dir, 'exact_shape_processing_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Exact Shape Cropping Summary\n")
            f.write(f"============================\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Successfully processed: {successful}\n")
            f.write(f"Failed: {len(image_files) - successful}\n")
            f.write(f"Success rate: {successful/len(image_files)*100:.1f}%\n")
            f.write(f"Settings: threshold={threshold}, padding={padding}\n")
            f.write(f"Edge smoothing: {edge_smoothing}, Feather edges: {feather_edges}\n\n")
            
            for result in results:
                if result.get('success'):
                    f.write(f"✅ {result['filename']} - Size: {result['cropped_size']}, "
                           f"Confidence: {result['confidence_score']:.3f}, "
                           f"Mask ratio: {result['mask_area_ratio']:.3f}\n")
                else:
                    f.write(f"❌ {result['filename']} - {result.get('error', 'Unknown error')}\n")
        
        print(f"\nExact shape processing complete! Summary saved to: {summary_path}")
        return results

# Usage examples:
# 
# # Single image processing with exact shape (transparent background)
# model = TubeSegmentationModel('deployment_package.pth')
# result = model.predict_and_crop_exact_shape('input_image.jpg', 'output_folder/', 
#                                           threshold=0.5, padding=5, 
#                                           transparent_background=True,
#                                           edge_smoothing=True, feather_edges=3)
# print(f"Exact shape cropped image saved to: {result['cropped_path']}")
#
# # Batch processing with exact shape cropping
# model = TubeSegmentationModel('deployment_package.pth')
# results = model.batch_process('input_folder/', 'output_folder/', 
#                              threshold=0.5, padding=10,
#                              transparent_background=True,
#                              edge_smoothing=True, feather_edges=2)
