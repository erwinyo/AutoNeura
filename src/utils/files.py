import os
import imghdr
import mimetypes

def is_media_file(file_path):
    """
    Check if the given file is an image or video.
    
    Args:
        file_path (str): Path to the file to check
        
    Returns:
        tuple: (is_media, media_type) where:
            - is_media (bool): True if file is image or video, False otherwise
            - media_type (str): 'image', 'video', or None if not media
    """
    if not os.path.isfile(file_path):
        return False, None
    
    # Initialize mimetypes
    mimetypes.init()
    
    # Check if it's an image file
    img_type = imghdr.what(file_path)
    if img_type is not None:
        return True, 'image'
    
    # Check if it's a video file
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('video/'):
        return True, 'video'
    
    # Additional check for common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    if any(file_path.lower().endswith(ext) for ext in video_extensions):
        return True, 'video'
    
    return False, None