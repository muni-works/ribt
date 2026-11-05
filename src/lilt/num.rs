//! English number words, written from scratch to reproduce the word sequence
//! num2words 0.5 (`lang='en'`) produces for misaki: cardinals with British
//! "and" after hundreds, ordinals, years, and floats read digit by digit
//! after "point". Output is already split the way misaki splits num2words'
//! text (`re.split(r'[^a-z]+')`), so "twenty-one" is `["twenty", "one"]`.
//! (num2words itself is LGPL; nothing here is copied from it.)

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [&str; 11] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
];

fn below_1000(n: u128, out: &mut Vec<String>) {
    debug_assert!(n < 1000);
    let (h, r) = (n / 100, n % 100);
    if h > 0 {
        out.push(ONES[h as usize].into());
        out.push("hundred".into());
        if r > 0 {
            out.push("and".into());
        }
    }
    if r == 0 && h > 0 {
        return;
    }
    if r < 20 {
        out.push(ONES[r as usize].into());
    } else {
        out.push(TENS[(r / 10) as usize].into());
        if r % 10 > 0 {
            out.push(ONES[(r % 10) as usize].into());
        }
    }
}

/// Cardinal words; `None` past the largest scale (num2words raises there).
pub fn cardinal(n: i128) -> Option<Vec<String>> {
    let mut out = Vec::new();
    if n < 0 {
        out.push("minus".into());
    }
    let mut m = n.unsigned_abs();
    if m == 0 {
        out.push("zero".into());
        return Some(out);
    }
    let mut groups = Vec::new();
    while m > 0 {
        groups.push(m % 1000);
        m /= 1000;
    }
    if groups.len() > SCALES.len() {
        return None;
    }
    let top = groups.len() - 1;
    for g in (0..groups.len()).rev() {
        let v = groups[g];
        if v == 0 {
            continue;
        }
        if g < top && g == 0 && v < 100 {
            out.push("and".into());
        }
        below_1000(v, &mut out);
        if g > 0 {
            out.push(SCALES[g].into());
        }
    }
    Some(out)
}

pub fn ordinal(n: i128) -> Option<Vec<String>> {
    let mut w = cardinal(n)?;
    let last = w.pop()?;
    let o = match last.as_str() {
        "one" => "first".to_string(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        s if s.ends_with('y') => format!("{}ieth", &s[..s.len() - 1]),
        s => format!("{s}th"),
    };
    w.push(o);
    Some(w)
}

pub fn year(n: i128) -> Option<Vec<String>> {
    let (v, bc) = (n.abs(), n < 0);
    let (high, low) = (v / 100, v % 100);
    let mut out = if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        cardinal(v)?
    } else {
        let mut o = cardinal(high)?;
        if low == 0 {
            o.push("hundred".into());
        } else if low < 10 {
            o.push("oh".into());
            o.extend(cardinal(low)?);
        } else {
            o.extend(cardinal(low)?);
        }
        o
    };
    if bc {
        out.push("bc".into()); // num2words appends "BC"
    }
    Some(out)
}

/// num2words(float(s)): integral values read as cardinals, otherwise
/// "<int> point d d d" over the shortest round-trip decimal digits.
pub fn float(s: &str) -> Option<Vec<String>> {
    let f: f64 = s.parse().ok()?;
    if !f.is_finite() {
        return None;
    }
    if f.fract() == 0.0 && f.abs() < 1e30 {
        return cardinal(f as i128);
    }
    let repr = format!("{f}");
    let (ip, fp) = repr.split_once('.')?;
    let pre: i128 = ip.parse().ok()?;
    let mut out = if pre == 0 && f < 0.0 { vec!["zero".to_string()] } else { cardinal(pre)? };
    out.push("point".into());
    for d in fp.chars() {
        out.push(ONES[d.to_digit(10)? as usize].into());
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn j(v: Option<Vec<String>>) -> String {
        v.unwrap().join(" ")
    }
    #[test]
    fn matches_num2words() {
        assert_eq!(
            j(cardinal(1234567)),
            "one million two hundred and thirty four thousand five hundred and sixty seven"
        );
        assert_eq!(j(cardinal(1001)), "one thousand and one");
        assert_eq!(j(cardinal(1100)), "one thousand one hundred");
        assert_eq!(j(cardinal(101000)), "one hundred and one thousand");
        assert_eq!(j(ordinal(21)), "twenty first");
        assert_eq!(j(ordinal(90)), "ninetieth");
        assert_eq!(j(ordinal(1000)), "one thousandth");
        assert_eq!(j(year(1905)), "nineteen oh five");
        assert_eq!(j(year(2005)), "two thousand and five");
        assert_eq!(j(year(2010)), "twenty ten");
        assert_eq!(j(year(1900)), "nineteen hundred");
        assert_eq!(j(float("3.14159")), "three point one four one five nine");
        assert_eq!(j(float("5.50")), "five point five");
        assert_eq!(j(float("1.0")), "one");
        assert_eq!(j(float("0.09")), "zero point zero nine");
    }
}
