from PIL import Image
import os 


def is_image(file_name:str) -> bool:
    """
    Check whether a file is a valid image.

    Parameters
    ----------
    file_name : str
        Path to the file.

    Returns
    -------
    bool
        True if file is a valid image, False otherwise.

    """
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False
    
def check_create_directory(path: str) -> None:    
    """
        Ensure that a directory exists; create it if it does not.

        Parameters
        ----------
        path : str
            Path of the directory to check or create.

        Returns
        -------
        None
            This function does not return anything.

        """

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")
