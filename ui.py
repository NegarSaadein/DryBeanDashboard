from shiny import ui
from components import (
    create_header,
    create_home,
    create_dataset,
    create_eda,
    create_statistics,
    create_preprocessing,
    create_models,
    create_evaluation
)

app_ui = ui.page_navbar(

    ui.nav_panel(
        "Home",
        create_home()
    ),

    ui.nav_panel(
        "Dataset",
        create_dataset()
    ),

    ui.nav_panel(
        "EDA",
        create_eda()
    ),

    ui.nav_panel(
        "Statistics",
        create_statistics()
    ),

    ui.nav_panel(
        "Preprocessing",
        create_preprocessing()
    ),

    ui.nav_panel(
        "Models",
        create_models()
    ),

    ui.nav_panel(
        "Evaluation",
        create_evaluation()
    ),

    title="Dry Bean Dashboard"
)