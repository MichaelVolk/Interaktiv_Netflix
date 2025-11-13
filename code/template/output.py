from IPython.display import HTML, display


def success(message="Diese Antwort ist richtig!"):
    display(HTML(f'''<div class="alert alert-success"> <i class="fas fa-check"></i> {message}</p></div>'''))


def wrong(message="Diese Antwort ist noch nicht richtig."):
    display(HTML(f'''<div class="alert alert-danger"> <i class="fas fa-times"></i> {message}</p></div>'''))


def html(html_source):
    display(HTML(html_source))
