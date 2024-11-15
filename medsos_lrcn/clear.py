import re

def clear_pattern(file_path):
    """
    Clears lines matching the pattern 'size of x:  torch.Size([8, 40, 512])' in a file.
    
    Args:
        file_path (str): The path to the file to process.
    """
    pattern = r"^size of x:  torch\.Size\(\[1, 40, 512\]\)\s*$"  # Regex pattern for the specific line
    #size of x:  torch.Size([1, 40, 512])
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Filter out lines that match the pattern
        filtered_lines = [line for line in lines if not re.match(pattern, line)]
        
        # Write the filtered lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(filtered_lines)
        
        print(f"Pattern removed successfully from {file_path}.")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# Replace 'your_file.txt' with the path to your target file
if __name__ == "__main__":
    file_path = 'medsos_log.txt'  # Update this with the file path
    clear_pattern(file_path)