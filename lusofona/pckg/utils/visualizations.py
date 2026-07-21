
from   IPython.display import display, HTML

# Plot dataframes side by side to avoid scrolling
def plot_dataframes_side_by_side( dfs:list, captions:list, tablespacing=5 ):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    for (caption, df) in zip(captions, dfs):
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += tablespacing * "\xa0"
    display(HTML(output))