import plotly.express as px


def bar_chart(
    dataframe,
    x,
    y,
    title,
    orientation="v",
):

    if dataframe.empty:
        return None

    return px.bar(
        dataframe,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        template="plotly_dark",
    )


def line_chart(
    dataframe,
    x,
    y,
    title,
):

    if dataframe.empty:
        return None

    return px.line(
        dataframe,
        x=x,
        y=y,
        title=title,
        markers=True,
        template="plotly_dark",
    )


def heatmap(
    dataframe,
    x,
    y,
    color,
    title,
):

    if dataframe.empty:
        return None

    return px.density_heatmap(
        dataframe,
        x=x,
        y=y,
        z=color,
        title=title,
        template="plotly_dark",
    )