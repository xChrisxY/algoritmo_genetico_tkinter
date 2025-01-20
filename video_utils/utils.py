import cv2
from pathlib import Path
import os

def save_frame(fig, frame_number, temp_dir, frame_files):
    """
    Guarda un frame individual de la animación como imagen.
    
    Args:
        fig: Figura de matplotlib
        frame_number: Número de frame
        temp_dir: Directorio temporal donde guardar los frames
        frame_files: Lista para almacenar las rutas de los frames
    """
    try:
        frame_path = temp_dir / f"frame_{frame_number:04d}.png"
        fig.savefig(str(frame_path), dpi=100, bbox_inches='tight')
        frame_files.append(frame_path)
        print(f"Saved frame {frame_number} to {frame_path}")
    except Exception as e:
        print(f"Error saving frame {frame_number}: {e}")

def generate_video(frame_files):
    """
    Genera un video a partir de los frames guardados.
    
    Args:
        frame_files: Lista de rutas a los archivos de frames
    """
    if not frame_files:
        print("No frames to generate video")
        return
        
    try:
        # Read the first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frame_files[0]}")
            
        height, width, _ = first_frame.shape
        
        output_path = 'evolucion_poblacion.mp4'
        
        # En Windows usar:
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        # En Linux/Mac usar:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            10.0,
            (width, height),
            isColor=True
        )

        # Ordenar los frames por número
        frame_files_sorted = sorted(frame_files, 
            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        
        # Write frames to video
        for frame_path in frame_files_sorted:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                print(f"Could not read frame: {frame_path}")
        
        out.release()
        print(f"Video saved as {output_path}")
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Video file created successfully. Size: {os.path.getsize(output_path)} bytes")
        else:
            print("Error: Video file was not created or is empty")
        
    except Exception as e:
        print(f"Error generating video: {e}")