"""The HTTP surface.

    app.py         the application factory and the error shapes
    auth.py        the signed persona header, and what is behind it
    deps.py        what every request resolves: layer, connection, user
    periods.py     a period label into the four dates the engine needs
    schemas.py     the response bodies the engine has no opinion about
    store.py       where a finished case lives between run and read
    streaming.py   the six SSE events
    routes/        the endpoints

Nothing in this package computes a business number. It resolves who is
asking, hands the request to `engine/`, and serialises what comes back.
"""
