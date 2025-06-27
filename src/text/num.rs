use std::collections::LinkedList;

#[derive(pest_derive::Parser)]
#[grammar = "resource/rule.pest"]
pub struct ExprParser;

pub mod zh {
    use crate::text::PhoneBuilder;

    use super::*;
    use pest::iterators::Pair;
    #[cfg(test)]
    use pest::Parser;

    fn parse_pn(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::pn);
        match pair.as_str() {
            "+" => builder.push_zh_word("加"),
            "-" => builder.push_zh_word("减"),
            "*" | "×" => builder.push_zh_word("乘"),
            "/" | "÷" => builder.push_zh_word("除以"),
            "=" => builder.push_zh_word("等于"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in pn", pair);
            }
        }
        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::flag);
        match pair.as_str() {
            "+" => builder.push_zh_word("正"),
            "-" => builder.push_zh_word("负"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in flag", pair);
            }
        }
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::percent);
        // percent = { (decimals|integer)~"%" }

        builder.push_zh_word("百分之");
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, builder)?,
                Rule::integer => {
                    parse_integer(pair, builder, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        Ok(())
    }

    static UNITS: [&str; 4] = ["", "十", "百", "千"];
    static BASE_UNITS: [&str; 4] = ["", "万", "亿", "万"];

    fn parse_integer(
        pair: Pair<Rule>,
        builder: &mut PhoneBuilder,
        unit: bool,
    ) -> anyhow::Result<LinkedList<(&'static str, &'static str)>> {
        assert_eq!(pair.as_rule(), Rule::integer);

        let mut r: LinkedList<(&str, &str)> = LinkedList::new();

        let inner = pair.into_inner().rev();
        let mut n = 0;

        for pair in inner {
            let txt = match pair.as_str() {
                "0" => "零",
                "1" => "一",
                "2" => "二",
                "3" => "三",
                "4" => "四",
                "5" => "五",
                "6" => "六",
                "7" => "七",
                "8" => "八",
                "9" => "九",
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in integer", n);
                    #[cfg(not(debug_assertions))]
                    ""
                }
            };
            let u = if n % 4 != 0 {
                UNITS[n % 4]
            } else {
                BASE_UNITS[(n / 4) % 4]
            };

            r.push_front((txt, u));
            n += 1;
        }

        if r.iter().all(|(s, _)| s == &"零") {
            builder.push_zh_word("零");
            return Ok(r);
        }

        if unit {
            let mut last_is_zero = true;
            for (s, u) in &r {
                if s == &"零" {
                    if !BASE_UNITS.contains(u) {
                        if !last_is_zero {
                            builder.push_zh_word(s);
                            last_is_zero = true;
                        }
                    } else {
                        builder.push_zh_word(u);
                    }
                } else {
                    builder.push_zh_word(s);
                    builder.push_zh_word(u);
                    last_is_zero = false;
                }
            }
        } else {
            for (s, _) in &r {
                builder.push_zh_word(s);
            }
        }

        Ok(r)
    }

    // cargo test --package gpt_sovits_rs --lib -- text::num::zh::test_parse_integer --exact --show-output
    #[test]
    fn test_parse_integer() {
        let mut builder = PhoneBuilder::new(false);

        let mut p = ExprParser::parse(Rule::integer, "0").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        let r = parse_integer(p, &mut builder, true).unwrap();
        for s in r {
            println!("{:?}", s);
        }

        println!("{:?}", builder.sentence.pop_back().unwrap());

        let mut p =
            ExprParser::parse(Rule::integer, "07801100170").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        let r = parse_integer(p, &mut builder, true).unwrap();
        for s in r {
            println!("{:?}", s);
        }

        println!("{:?}", builder.sentence.pop_back().unwrap());

        let mut p = ExprParser::parse(Rule::integer, "1010").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        let r = parse_integer(p, &mut builder, true).unwrap();
        for s in r {
            println!("{:?}", s);
        }

        println!("{:?}", builder.sentence.pop_back().unwrap());

        let mut builder = PhoneBuilder::new(false);

        let mut p =
            ExprParser::parse(Rule::integer, "034056009009040").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        let r = parse_integer(p, &mut builder, false).unwrap();
        for s in r {
            println!("{:?}", s);
        }

        println!("{:?}", builder.sentence.pop_back().unwrap());
    }

    fn parse_decimals(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::decimals);

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, builder, true)?;
        } else {
            builder.push_zh_word("零");
        }
        builder.push_zh_word("点");
        parse_integer(f_part, builder, false)?;

        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::fractional);

        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(denominator, builder, true)?;
        builder.push_zh_word("分之");
        parse_integer(numerator, builder, true)?;

        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::num);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, builder)?,
                Rule::percent => parse_percent(pair, builder)?,
                Rule::decimals => {
                    parse_decimals(pair, builder)?;
                }
                Rule::fractional => {
                    parse_fractional(pair, builder)?;
                }
                Rule::integer => {
                    parse_integer(pair, builder, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in num", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::signs);

        let inner = pair.into_inner();
        for pair in inner {
            log::debug!("{:?}", pair);
            match pair.as_rule() {
                Rule::num => parse_num(pair, builder)?,
                Rule::pn => parse_pn(pair, builder)?,
                Rule::word => {
                    log::warn!("word: {:?}", pair.as_str());
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_signs() {
        let mut builder = PhoneBuilder::new(false);

        let mut p = ExprParser::parse(Rule::signs, "034.9/(9.2)+ -89.2%=99/10")
            .unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_signs(p, &mut builder).unwrap();

        println!("{:?}", builder.sentence.back().unwrap());
    }

    #[test]
    fn test_parser() {
        let pairs = ExprParser::parse(Rule::all, "1+2=3.65 ").unwrap_or_else(|e| panic!("{}", e));
        if let Some(pair) = pairs.last() {
            let pairs = pair.into_inner();
            for pair in pairs {
                println!("==={:?}", pair);
            }
        }
    }

    fn parse_link(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::link);

        match pair.as_str() {
            "-" => builder.push_zh_word("杠"),
            _ => builder.push_punctuation("…"),
        }

        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "零",
                        "1" => "一",
                        "2" => "二",
                        "3" => "三",
                        "4" => "四",
                        "5" => "五",
                        "6" => "六",
                        "7" => "七",
                        "8" => "八",
                        "9" => "九",
                        n => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in integer", n);

                            #[cfg(not(debug_assertions))]
                            n
                        }
                    };
                    builder.push_zh_word(txt);
                }
                Rule::alpha => {
                    let txt = pair.as_str();
                    builder.push_en_word(txt);
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "阿尔法",
                        "β" | "Β" => "贝塔",
                        "γ" | "Γ" => "伽马",
                        "δ" | "Δ" => "德尔塔",
                        "ε" | "Ε" => "艾普西龙",
                        "ζ" | "Ζ" => "泽塔",
                        "η" | "Η" => "艾塔",
                        "θ" | "Θ" => "西塔",
                        "ι" | "Ι" => "约塔",
                        "κ" | "Κ" => "卡帕",
                        "λ" | "Λ" => "兰姆达",
                        "μ" | "Μ" => "缪",
                        "ν" | "Ν" => "纽",
                        "ξ" | "Ξ" => "克西",
                        "ο" | "Ο" => "欧米克戈",
                        "π" | "Π" => "派",
                        "ρ" | "Ρ" => "罗",
                        "σ" | "Σ" => "西格玛",
                        "τ" | "Τ" => "套",
                        "υ" | "Υ" => "宇普西龙",
                        "φ" | "Φ" => "斐",
                        "χ" | "Χ" => "希",
                        "ψ" | "Ψ" => "普西",
                        "ω" | "Ω" => "欧米伽",
                        _ => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in greek", pair.as_str());
                            #[cfg(not(debug_assertions))]
                            ""
                        }
                    };
                    builder.push_zh_word(txt);
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in word", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::ident);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, builder)?,
                Rule::link => parse_link(pair, builder)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in ident", pair.as_str());
                }
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::all);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, builder)?,
                Rule::ident => parse_ident(pair, builder)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in all", pair.as_str());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_parse_all() {
        let mut builder = PhoneBuilder::new(false);

        let mut p =
            ExprParser::parse(Rule::all, "1.6+2.005=3.65%").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_all(p, &mut builder).unwrap();
        match builder.sentence.back().unwrap() {
            crate::text::Sentence::Zh(zh) => {
                println!("{:?}", zh.zh_text);
            }
            _ => {
                #[cfg(debug_assertions)]
                unreachable!();
            }
        }
    }

    #[test]
    fn test_parse_all_ident() {
        let mut builder = PhoneBuilder::new(false);

        let mut p = ExprParser::parse(Rule::all, "GPT-α96").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_all(p, &mut builder).unwrap();
        println!("{:?}", builder.sentence);
    }
}

pub mod en {
    use crate::text::PhoneBuilder;

    use super::super::SEPARATOR;
    use super::*;
    use pest::iterators::Pair;
    #[cfg(test)]
    use pest::Parser;

    fn parse_pn(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::pn);
        match pair.as_str() {
            "+" => builder.push_en_word("plus"),
            "-" => builder.push_en_word("minus"),
            "*" | "×" => builder.push_en_word("times"),
            "/" | "÷" => {
                builder.push_en_word("divided");
                builder.push_punctuation(SEPARATOR);
                builder.push_en_word("by");
            }
            "=" => builder.push_en_word("is"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in pn", pair);
            }
        }
        builder.push_punctuation(SEPARATOR);

        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::flag);
        match pair.as_str() {
            "-" => builder.push_en_word("negative"),
            _ => {
                #[cfg(debug_assertions)]
                unreachable!("unknown: {:?} in flag", pair);
            }
        }
        builder.push_punctuation(SEPARATOR);
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::percent);
        // percent = { (decimals|integer)~"%" }

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, builder)?,
                Rule::integer => {
                    parse_integer(pair, builder, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        builder.push_en_word("percent");
        builder.push_punctuation(SEPARATOR);

        Ok(())
    }

    fn parse_integer(
        pair: Pair<Rule>,
        builder: &mut PhoneBuilder,
        unit: bool,
    ) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::integer);
        if unit {
            if let Ok(r) = num2en::str_to_words(pair.as_str()) {
                r.split(&[' ', '-']).for_each(|s| {
                    builder.push_en_word(s);
                    builder.push_punctuation(SEPARATOR);
                });
                return Ok(());
            }
        }

        let inner = pair.into_inner();
        for pair in inner {
            let txt = match pair.as_str() {
                "0" => "zero",
                "1" => "one",
                "2" => "two",
                "3" => "three",
                "4" => "four",
                "5" => "five",
                "6" => "six",
                "7" => "seven",
                "8" => "eight",
                "9" => "nine",
                n => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in integer", n);
                    #[cfg(not(debug_assertions))]
                    n
                }
            };
            builder.push_en_word(txt);
            builder.push_punctuation(SEPARATOR);
        }

        Ok(())
    }

    #[test]
    fn test_parse_integer() {
        let mut builder = PhoneBuilder::new(false);

        let mut p =
            ExprParser::parse(Rule::integer, "034056009009040").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_integer(p, &mut builder, true).unwrap();

        println!("{:?}", builder.sentence.back().unwrap());

        let mut builder = PhoneBuilder::new(false);

        let mut p =
            ExprParser::parse(Rule::integer, "034056009009040").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_integer(p, &mut builder, false).unwrap();

        println!("{:?}", builder.sentence.back().unwrap());
    }

    fn parse_decimals(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::decimals);
        if let Ok(r) = num2en::str_to_words(pair.as_str()) {
            r.split(&[' ', '-']).for_each(|s| {
                builder.push_en_word(s);
                builder.push_punctuation(SEPARATOR);
            });
            return Ok(());
        }

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, builder, true)?;
        } else {
            builder.push_en_word("zero");
            builder.push_punctuation(SEPARATOR);
        }
        builder.push_en_word("point");
        builder.push_punctuation(SEPARATOR);

        parse_integer(f_part, builder, false)?;

        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::fractional);

        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(numerator, builder, true)?;
        builder.push_en_word("over");
        builder.push_punctuation(SEPARATOR);
        parse_integer(denominator, builder, true)?;

        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::num);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, builder)?,
                Rule::percent => parse_percent(pair, builder)?,
                Rule::decimals => {
                    parse_decimals(pair, builder)?;
                }
                Rule::fractional => {
                    parse_fractional(pair, builder)?;
                }
                Rule::integer => {
                    parse_integer(pair, builder, true)?;
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in num", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::signs);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::num => parse_num(pair, builder)?,
                Rule::pn => parse_pn(pair, builder)?,
                Rule::word => {
                    println!("word: {:?}", pair.as_str());
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in expr", pair.as_str());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_signs() {
        let mut builder = PhoneBuilder::new(false);

        let mut p = ExprParser::parse(Rule::signs, "034.9/(9.2)+ -89.2%=99/10")
            .unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_signs(p, &mut builder).unwrap();

        println!("{:?}", builder.sentence.back().unwrap());
    }

    #[test]
    fn test_parser() {
        let pairs = ExprParser::parse(Rule::all, "1+2=3.65 ").unwrap_or_else(|e| panic!("{}", e));
        if let Some(pair) = pairs.last() {
            let pairs = pair.into_inner();
            for pair in pairs {
                println!("==={:?}", pair);
            }
        }
    }

    fn parse_link(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::link);

        builder.push_punctuation(super::super::SEPARATOR);

        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "zero",
                        "1" => "one",
                        "2" => "two",
                        "3" => "three",
                        "4" => "four",
                        "5" => "five",
                        "6" => "six",
                        "7" => "seven",
                        "8" => "eight",
                        "9" => "nine",
                        n => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in integer", n);
                            #[cfg(not(debug_assertions))]
                            n
                        }
                    };
                    builder.push_en_word(txt);
                    builder.push_punctuation(SEPARATOR);
                }
                Rule::alpha => {
                    let txt = pair.as_str();
                    builder.push_en_word(txt);
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "alpha",
                        "β" | "Β" => "beta",
                        "γ" | "Γ" => "gamma",
                        "δ" | "Δ" => "delta",
                        "ε" | "Ε" => "epsilon",
                        "ζ" | "Ζ" => "zeta",
                        "η" | "Η" => "eta",
                        "θ" | "Θ" => "theta",
                        "ι" | "Ι" => "iota",
                        "κ" | "Κ" => "kappa",
                        "λ" | "Λ" => "lambda",
                        "μ" | "Μ" => "mu",
                        "ν" | "Ν" => "nu",
                        "ξ" | "Ξ" => "xi",
                        "ο" | "Ο" => "omicron",
                        "π" | "Π" => "pi",
                        "ρ" | "Ρ" => "rho",
                        "σ" | "Σ" => "sigma",
                        "τ" | "Τ" => "tau",
                        "υ" | "Υ" => "upsilon",
                        "φ" | "Φ" => "phi",
                        "χ" | "Χ" => "chi",
                        "ψ" | "Ψ" => "psi",
                        "ω" | "Ω" => "omega",

                        _ => {
                            #[cfg(debug_assertions)]
                            unreachable!("unknown: {:?} in greek", pair.as_str());
                            #[cfg(not(debug_assertions))]
                            ""
                        }
                    };
                    builder.push_en_word(txt);
                    builder.push_punctuation(SEPARATOR);
                }
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in word", pair.as_str());
                }
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::ident);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, builder)?,
                Rule::link => parse_link(pair, builder)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in ident", pair.as_str());
                }
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, builder: &mut PhoneBuilder) -> anyhow::Result<()> {
        assert_eq!(pair.as_rule(), Rule::all);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, builder)?,
                Rule::ident => parse_ident(pair, builder)?,
                _ => {
                    #[cfg(debug_assertions)]
                    unreachable!("unknown: {:?} in all", pair.as_str());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_parse_all() {
        let mut builder = PhoneBuilder::new(false);

        let mut p =
            ExprParser::parse(Rule::all, "1.6+2.005=3.65%").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_all(p, &mut builder).unwrap();
        match builder.sentence.back().unwrap() {
            crate::text::Sentence::En(en) => {
                println!("{:?}", en.en_text);
            }
            _ => {
                #[cfg(debug_assertions)]
                unreachable!();
            }
        }
    }

    #[test]
    fn test_parse_all_ident() {
        let mut builder = PhoneBuilder::new(false);

        let mut p = ExprParser::parse(Rule::all, "GPT-α96").unwrap_or_else(|e| panic!("{}", e));
        let p = p.next().unwrap();
        parse_all(p, &mut builder).unwrap();
        println!("{:?}", builder.sentence);
    }
}
