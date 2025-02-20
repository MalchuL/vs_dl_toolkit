import pytweening


def interpolate(alpha, method):
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    out = vars(pytweening)[method](alpha)
    return out

