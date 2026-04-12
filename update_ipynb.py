import nbformat

def main():
    md_file = "C:/Users/saksd/Desktop/NJIT/Coursework/Computer-Vision-Course/assignment-4/assignment-4.md"
    ipynb_file = "C:/Users/saksd/Desktop/NJIT/Coursework/Computer-Vision-Course/assignment-4/assignment-4.ipynb"
    
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell(md_content))
    
    with open(ipynb_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
if __name__ == "__main__":
    main()
