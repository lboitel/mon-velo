"""French public holidays (fixed dates + movable dates computed from Easter)."""
from datetime import date, timedelta


def easter(year):
    # Meeus/Jones/Butcher algorithm (Gregorian calendar)
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def holidays_for_year(year):
    e = easter(year)
    return {
        date(year, 1, 1),     # New Year's Day
        e + timedelta(days=1),    # Easter Monday
        date(year, 5, 1),     # Labour Day
        date(year, 5, 8),     # WWII Victory Day
        e + timedelta(days=39),   # Ascension Day
        e + timedelta(days=50),   # Whit Monday
        date(year, 7, 14),    # Bastille Day
        date(year, 8, 15),    # Assumption Day
        date(year, 11, 1),    # All Saints' Day
        date(year, 11, 11),   # Armistice Day
        date(year, 12, 25),   # Christmas Day
    }


_holiday_cache = {}


def is_holiday(d):
    """d: datetime.date"""
    if d.year not in _holiday_cache:
        _holiday_cache[d.year] = holidays_for_year(d.year)
    return d in _holiday_cache[d.year]
