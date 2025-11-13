import ipywidgets as widgets
from IPython.display import display
import template.output as output
import template.widget_state_storage as storage
import html


WIDGET_TYPE = "Freitext"


def load_answer(identifier):
    state = storage.load_state(WIDGET_TYPE, identifier, None)
    if state is None:
        return None

    return state["Antwort"]


def store_answer(identifier, input_description, answer):
    state = {
        "Beschreibung": input_description,
        "Antwort": answer
    }
    storage.store_state(WIDGET_TYPE, identifier, state)


def prompt_answer(exercise_identifier, input_prompt="Die Frage:", input_description="Die Antwort:"):
    existing_answer = load_answer(exercise_identifier)

    # Define widgets
    output = widgets.Output()

    input_area = widgets.Textarea(
        placeholder='Gebe hier die Antwort ein und bestätige anschließend die Eingabe.',
        description=f'{input_prompt}:',
        disabled=existing_answer is not None,
        value=existing_answer,
        rows=2,
        style=dict(description_width='initial'),
        layout=widgets.Layout(width='75%')
    )

    def to_save_button(button):
        button.description = "Speichern"
        button.icon = "save"

    def to_edit_button(button):
        button.description = "Bearbeiten"
        button.icon = "edit"

    button = widgets.Button()

    if existing_answer is None:
        to_save_button(button)
    else:
        to_edit_button(button)

        with output:
            output.clear_output()
            _beautify_output(input_description, input_area.value)

    # Handle click logic
    def handle_click(button):
        if button.description == "Speichern":
            input_area.disabled = True
            to_edit_button(button)

            with output:
                output.clear_output()

                _beautify_output(input_description, input_area.value)

            store_answer(exercise_identifier, input_description, input_area.value)
        else:
            input_area.disabled = False
            to_save_button(button)

    button.on_click(handle_click)

    # Show widget
    widget = widgets.HBox([input_area, button])
    display(widget)
    display(output)


def _beautify_output(label, answer):
    html_source = """
    <div id="student-answer" class="alert alert-info">
      <h4>{label}</h4>
      {answer}
    </div>
    """.format(label=label, answer=html.escape(answer).replace("\n", "<br>"))

    output.html(html_source)
    

def print_answer(identifier):
    state = storage.load_state(WIDGET_TYPE, identifier, None)
    if state is None:
        print("Die Antwort wurde nicht gefunden.")
        return

    _beautify_output(state["Beschreibung"], state["Antwort"])
