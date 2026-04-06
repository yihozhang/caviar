//! Stochastic rewrite rules for the `Math` language.
//!
//! Each deterministic rule in `rules/` has a corresponding entry here.
//! Conditional rules use [`StoConditionalApplier`] with condition closures
//! that inspect [`StoData`] (constant-folded values) instead of an e-graph.

use std::sync::Arc;

use smallvec::SmallVec;

use egg::stochastic::{
    State, StoApplier, StoConditionalApplier, StoRewrite, StoSearchMatch, StoSearcher,
};
use egg::{Id, Language, Pattern, RecExpr, Subst, Var};

use crate::trs::Math;

// ─── Analysis ────────────────────────────────────────────────────────────────

/// Per-node analysis data for stochastic rewriting: the constant value of the
/// node (if it can be fully evaluated), used both for constant folding and for
/// the cost/condition helpers below.
#[derive(Debug, Clone)]
pub struct StoData {
    pub constant: Option<i64>,
}

/// The stochastic analysis for [`Math`].
///
/// It computes constant folding bottom-up (mirroring [`crate::trs::ConstantFold`])
/// and assigns cost 0 to the constants `0` and `1` (our proof goals) so that
/// the Metropolis-Hastings sampler is incentivised to prove expressions.
#[derive(Default, Clone)]
pub struct StoConstantFold;

impl egg::stochastic::StoAnalysis<Math> for StoConstantFold {
    type Data = StoData;

    fn make(&self, enode: &Math, analysis: &[StoData]) -> StoData {
        let x = |id: &Id| analysis[usize::from(*id)].constant;
        let constant = match enode {
            Math::Constant(c) => Some(*c),
            Math::Add([a, b]) => x(a).and_then(|va| x(b).map(|vb| va.wrapping_add(vb))),
            Math::Sub([a, b]) => x(a).and_then(|va| x(b).map(|vb| va.wrapping_sub(vb))),
            Math::Mul([a, b]) => x(a).and_then(|va| x(b).map(|vb| va.wrapping_mul(vb))),
            Math::Div([a, b]) => x(a).and_then(|va| {
                x(b).filter(|&vb| vb != 0).map(|vb| va / vb)
            }),
            Math::Mod([a, b]) => x(a).and_then(|va| {
                x(b).filter(|&vb| vb != 0).map(|vb| va % vb)
            }),
            Math::Max([a, b]) => x(a).and_then(|va| x(b).map(|vb| std::cmp::max(va, vb))),
            Math::Min([a, b]) => x(a).and_then(|va| x(b).map(|vb| std::cmp::min(va, vb))),
            Math::Not(a) => x(a).map(|va| if va == 0 { 1 } else { 0 }),
            Math::Lt([a, b]) => x(a).and_then(|va| x(b).map(|vb| if va < vb { 1 } else { 0 })),
            Math::Gt([a, b]) => x(a).and_then(|va| x(b).map(|vb| if va > vb { 1 } else { 0 })),
            Math::Let([a, b]) => x(a).and_then(|va| x(b).map(|vb| if va <= vb { 1 } else { 0 })),
            Math::Get([a, b]) => x(a).and_then(|va| x(b).map(|vb| if va >= vb { 1 } else { 0 })),
            Math::Eq([a, b]) => x(a).and_then(|va| x(b).map(|vb| if va == vb { 1 } else { 0 })),
            Math::IEq([a, b]) => x(a).and_then(|va| x(b).map(|vb| if va != vb { 1 } else { 0 })),
            Math::And([a, b]) => x(a).and_then(|va| {
                x(b).map(|vb| if va == 0 || vb == 0 { 0 } else { 1 })
            }),
            Math::Or([a, b]) => x(a).and_then(|va| {
                x(b).map(|vb| if va == 1 || vb == 1 { 1 } else { 0 })
            }),
            Math::Symbol(_) => None,
        };
        StoData { constant }
    }

    /// Cost is **0** for the proof goals `0` and `1`; AST size otherwise.
    fn cost(&self, enode: &Math, _analysis: &[StoData], children_cost: &[f64]) -> f64 {
        match enode {
            Math::Constant(0) | Math::Constant(1) => 0.0,
            _ => 1.0 + enode.fold(0.0, |acc, id| acc + children_cost[usize::from(id)]),
        }
    }

    /// Perform constant folding: replace a node by its computed constant.
    fn modify(&self, state: &mut State<Math, Self>, pos: Id) {
        if let Some(c) = state.analysis[usize::from(pos)].constant {
            let folded = Math::Constant(c);
            if state.rec_expr[pos] != folded {
                state.rec_expr[pos] = folded;
                state.touch(pos);
            }
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

type StoRw = StoRewrite<Math, StoConstantFold>;
type St = State<Math, StoConstantFold>;

fn v(s: &str) -> Var {
    s.parse().unwrap()
}

fn pat(s: &str) -> Pattern<Math> {
    s.parse().unwrap()
}

fn sto_rw(name: &str, lhs: &str, rhs: &str) -> StoRw {
    StoRewrite::new(name, pat(lhs), pat(rhs)).unwrap()
}

fn sto_rw_cond<F>(name: &str, lhs: &str, rhs: &str, cond: F) -> StoRw
where
    F: Fn(&St, Id, &Subst) -> bool + Send + Sync + 'static,
{
    let applier = StoConditionalApplier {
        applier: Arc::new(pat(rhs)),
        condition: Box::new(cond),
    };
    StoRewrite::new(name, pat(lhs), applier).unwrap()
}

// ─── Condition helpers ────────────────────────────────────────────────────────

fn const_of(state: &St, id: Id) -> Option<i64> {
    state.analysis[usize::from(id)].constant
}

fn sto_is_const_pos(
    var: &'static str,
) -> impl Fn(&St, Id, &Subst) -> bool + Send + Sync + 'static {
    move |state, _, subst| matches!(const_of(state, subst[v(var)]), Some(c) if c > 0)
}

fn sto_is_const_neg(
    var: &'static str,
) -> impl Fn(&St, Id, &Subst) -> bool + Send + Sync + 'static {
    move |state, _, subst| matches!(const_of(state, subst[v(var)]), Some(c) if c < 0)
}

fn sto_is_not_zero(
    var: &'static str,
) -> impl Fn(&St, Id, &Subst) -> bool + Send + Sync + 'static {
    move |state, _, subst| !matches!(const_of(state, subst[v(var)]), Some(0))
}

fn sto_compare_c0_c1(
    var: &'static str,
    var1: &'static str,
    comp: &'static str,
) -> impl Fn(&St, Id, &Subst) -> bool + Send + Sync + 'static {
    move |state, _, subst| {
        match (const_of(state, subst[v(var)]), const_of(state, subst[v(var1)])) {
            (Some(c), Some(c1)) => match comp {
                "<" => c < c1,
                "<a" => c < c1.abs(),
                "<=" => c <= c1,
                "<=+1" => c <= c1 + 1,
                "<=a" => c <= c1.abs(),
                "<=-a" => c <= -c1.abs(),
                "<=-a+1" => c <= 1 - c1.abs(),
                ">" => c > c1,
                ">a" => c > c1.abs(),
                ">=" => c >= c1,
                ">=a" => c >= c1.abs(),
                ">=a-1" => c >= c1.abs() - 1,
                "!=" => c != c1,
                "%0" => c1 != 0 && c % c1 == 0,
                "!%0" => c1 != 0 && c % c1 != 0,
                "%0<" => c1 > 0 && c % c1 == 0,
                "%0>" => c1 < 0 && c % c1 == 0,
                _ => false,
            },
            _ => false,
        }
    }
}

// ─── Rule sets ───────────────────────────────────────────────────────────────

pub fn sto_add() -> Vec<StoRw> {
    vec![
        sto_rw("add-comm", "(+ ?a ?b)", "(+ ?b ?a)"),
        sto_rw("add-assoc", "(+ ?a (+ ?b ?c))", "(+ (+ ?a ?b) ?c)"),
        sto_rw("add-zero", "(+ ?a 0)", "?a"),
        sto_rw("add-dist-mul", "(* ?a (+ ?b ?c))", "(+ (* ?a ?b) (* ?a ?c))"),
        sto_rw("add-fact-mul", "(+ (* ?a ?b) (* ?a ?c))", "(* ?a (+ ?b ?c))"),
        sto_rw("add-denom-mul", "(+ (/ ?a ?b) ?c)", "(/ (+ ?a (* ?b ?c)) ?b)"),
        sto_rw("add-denom-div", "(/ (+ ?a (* ?b ?c)) ?b)", "(+ (/ ?a ?b) ?c)"),
        sto_rw(
            "add-div-mod",
            "( + ( / ?x 2 ) ( % ?x 2 ) )",
            "( / ( + ?x 1 ) 2 )",
        ),
        sto_rw_cond(
            "add-const",
            "( + (* ?x ?a) (* ?y ?b))",
            "( * (+ (* ?x (/ ?a ?b)) ?y) ?b)",
            sto_compare_c0_c1("?a", "?b", "%0"),
        ),
    ]
}

pub fn sto_mul() -> Vec<StoRw> {
    vec![
        sto_rw("mul-comm", "(* ?a ?b)", "(* ?b ?a)"),
        sto_rw("mul-assoc", "(* ?a (* ?b ?c))", "(* (* ?a ?b) ?c)"),
        sto_rw("mul-zero", "(* ?a 0)", "0"),
        sto_rw("mul-one", "(* ?a 1)", "?a"),
        sto_rw("mul-cancel-div", "(* (/ ?a ?b) ?b)", "(- ?a (% ?a ?b))"),
        sto_rw("mul-max-min", "(* (max ?a ?b) (min ?a ?b))", "(* ?a ?b)"),
        sto_rw("div-cancel-mul", "(/ (* ?y ?x) ?x)", "?y"),
    ]
}

pub fn sto_sub() -> Vec<StoRw> {
    vec![sto_rw("sub-to-add", "(- ?a ?b)", "(+ ?a (* -1 ?b))")]
}

pub fn sto_div() -> Vec<StoRw> {
    vec![
        sto_rw("div-zero", "(/ 0 ?x)", "(0)"),
        sto_rw_cond("div-cancel", "(/ ?a ?a)", "1", sto_is_not_zero("?a")),
        sto_rw("div-minus-down", "(/ (* -1 ?a) ?b)", "(/ ?a (* -1 ?b))"),
        sto_rw("div-minus-up", "(/ ?a (* -1 ?b))", "(/ (* -1 ?a) ?b)"),
        sto_rw("div-minus-in", "(* -1 (/ ?a ?b))", "(/ (* -1 ?a) ?b)"),
        sto_rw("div-minus-out", "(/ (* -1 ?a) ?b)", "(* -1 (/ ?a ?b))"),
        sto_rw_cond(
            "div-consts-div",
            "( / ( * ?x ?a ) ?b )",
            "( / ?x ( / ?b ?a ) )",
            sto_compare_c0_c1("?b", "?a", "%0<"),
        ),
        sto_rw_cond(
            "div-consts-mul",
            "( / ( * ?x ?a ) ?b )",
            "( * ?x ( / ?a ?b ) )",
            sto_compare_c0_c1("?a", "?b", "%0<"),
        ),
        sto_rw_cond(
            "div-consts-add",
            "( / ( + ( * ?x ?a ) ?y ) ?b )",
            "( + ( * ?x ( / ?a ?b ) ) ( / ?y ?b ) )",
            sto_compare_c0_c1("?a", "?b", "%0<"),
        ),
        sto_rw_cond(
            "div-separate",
            "( / ( + ?x ?a ) ?b )",
            "( + ( / ?x ?b ) ( / ?a ?b ) )",
            sto_compare_c0_c1("?a", "?b", "%0<"),
        ),
    ]
}

pub fn sto_eq() -> Vec<StoRw> {
    vec![
        sto_rw("eq-comm", "(== ?x ?y)", "(== ?y ?x)"),
        sto_rw("eq-x-y-0", "(== ?x ?y)", "(== (- ?x ?y) 0)"),
        sto_rw("eq-swap", "(== (+ ?x ?y) ?z)", "(== ?x (- ?z ?y))"),
        sto_rw("eq-x-x", "(== ?x ?x)", "1"),
        sto_rw("eq-mul-x-y-0", "(== (* ?x ?y) 0)", "(|| (== ?x 0) (== ?y 0))"),
        sto_rw("eq-max-lt", "( == (max ?x ?y) ?y)", "(<= ?x ?y)"),
        sto_rw("Eq-min-lt", "( == (min ?x ?y) ?y)", "(<= ?y ?x)"),
        sto_rw("Eq-lt-min", "(<= ?y ?x)", "( == (min ?x ?y) ?y)"),
        sto_rw_cond(
            "Eq-a-b",
            "(== (* ?a ?x) ?b)",
            "0",
            sto_compare_c0_c1("?b", "?a", "!%0"),
        ),
        sto_rw_cond(
            "Eq-max-c-pos",
            "(== (max ?x ?c) 0)",
            "0",
            sto_is_const_pos("?c"),
        ),
        sto_rw_cond(
            "Eq-max-c-neg",
            "(== (max ?x ?c) 0)",
            "(== ?x 0)",
            sto_is_const_neg("?c"),
        ),
        sto_rw_cond(
            "Eq-min-c-pos",
            "(== (min ?x ?c) 0)",
            "0",
            sto_is_const_neg("?c"),
        ),
        sto_rw_cond(
            "Eq-min-c-neg",
            "(== (min ?x ?c) 0)",
            "(== ?x 0)",
            sto_is_const_pos("?c"),
        ),
    ]
}

pub fn sto_and() -> Vec<StoRw> {
    vec![
        sto_rw("and-comm", "(&& ?y ?x)", "(&& ?x ?y)"),
        sto_rw("and-assoc", "(&& ?a (&& ?b ?c))", "(&& (&& ?a ?b) ?c)"),
        sto_rw("and-x-1", "(&& 1 ?x)", "?x"),
        sto_rw("and-x-x", "(&& ?x ?x)", "?x"),
        sto_rw("and-x-not-x", "(&& ?x (! ?x))", "0"),
        sto_rw_cond(
            "and-eq-eq",
            "( && ( == ?x ?c0 ) ( == ?x ?c1 ) )",
            "0",
            sto_compare_c0_c1("?c1", "?c0", "!="),
        ),
        sto_rw_cond(
            "and-ineq-eq",
            "( && ( != ?x ?c0 ) ( == ?x ?c1 ) )",
            "( == ?x ?c1 )",
            sto_compare_c0_c1("?c1", "?c0", "!="),
        ),
        sto_rw("and-lt-to-min", "(&& (< ?x ?y) (< ?x ?z))", "(< ?x (min ?y ?z))"),
        sto_rw("and-min-to-lt", "(< ?x (min ?y ?z))", "(&& (< ?x ?y) (< ?x ?z))"),
        sto_rw("and-eqlt-to-min", "(&& (<= ?x ?y) (<= ?x ?z))", "(<= ?x (min ?y ?z))"),
        sto_rw("and-min-to-eqlt", "(<= ?x (min ?y ?z))", "(&& (<= ?x ?y) (<= ?x ?z))"),
        sto_rw("and-lt-to-max", "(&& (< ?y ?x) (< ?z ?x))", "(< (max ?y ?z) ?x)"),
        sto_rw("and-max-to-lt", "(> ?x (max ?y ?z))", "(&& (< ?z ?x) (< ?y ?x))"),
        sto_rw("and-eqlt-to-max", "(&& (<= ?y ?x) (<= ?z ?x))", "(<= (max ?y ?z) ?x)"),
        sto_rw("and-max-to-eqlt", "(>= ?x (max ?y ?z))", "(&& (<= ?z ?x) (<= ?y ?x))"),
        sto_rw_cond(
            "and-lt-gt-to-0",
            "( && ( < ?c0 ?x ) ( < ?x ?c1 ) )",
            "0",
            sto_compare_c0_c1("?c1", "?c0", "<=+1"),
        ),
        sto_rw_cond(
            "and-eqlt-eqgt-to-0",
            "( && ( <= ?c0 ?x ) ( <= ?x ?c1 ) )",
            "0",
            sto_compare_c0_c1("?c1", "?c0", "<"),
        ),
        sto_rw_cond(
            "and-eqlt-gt-to-0",
            "( && ( <= ?c0 ?x ) ( < ?x ?c1 ) )",
            "0",
            sto_compare_c0_c1("?c1", "?c0", "<="),
        ),
    ]
}

pub fn sto_or() -> Vec<StoRw> {
    vec![
        sto_rw("or-to-and", "(|| ?x ?y)", "(! (&& (! ?x) (! ?y)))"),
        sto_rw("or-comm", "(|| ?y ?x)", "(|| ?x ?y)"),
    ]
}

pub fn sto_not() -> Vec<StoRw> {
    vec![
        sto_rw("eqlt-to-not-gt", "(<= ?x ?y)", "(! (< ?y ?x))"),
        sto_rw("not-gt-to-eqlt", "(! (< ?y ?x))", "(<= ?x ?y)"),
        sto_rw("eqgt-to-not-lt", "(>= ?x ?y)", "(! (< ?x ?y))"),
        sto_rw("not-eq-to-ineq", "(! (== ?x ?y))", "(!= ?x ?y)"),
        sto_rw("not-not", "(! (! ?x))", "?x"),
    ]
}

pub fn sto_max() -> Vec<StoRw> {
    vec![sto_rw(
        "max-to-min",
        "(max ?a ?b)",
        "(* -1 (min (* -1 ?a) (* -1 ?b)))",
    )]
}

pub fn sto_min() -> Vec<StoRw> {
    vec![
        sto_rw("min-comm", "(min ?a ?b)", "(min ?b ?a)"),
        sto_rw("min-ass", "(min (min ?x ?y) ?z)", "(min ?x (min ?y ?z))"),
        sto_rw("min-x-x", "(min ?x ?x)", "?x"),
        sto_rw("min-max", "(min (max ?x ?y) ?x)", "?x"),
        sto_rw(
            "min-max-max-x",
            "(min (max ?x ?y) (max ?x ?z))",
            "(max (min ?y ?z) ?x)",
        ),
        sto_rw(
            "min-max-min-y",
            "(min (max (min ?x ?y) ?z) ?y)",
            "(min (max ?x ?z) ?y)",
        ),
        sto_rw("min-sub-both", "(min (+ ?a ?b) ?c)", "(+ (min ?b (- ?c ?a)) ?a)"),
        sto_rw("min-add-both", "(+ (min ?x ?y) ?z)", "(min (+ ?x ?z) (+ ?y ?z))"),
        sto_rw_cond(
            "min-x-x-plus-a-pos",
            "(min ?x (+ ?x ?a))",
            "?x",
            sto_is_const_pos("?a"),
        ),
        sto_rw_cond(
            "min-x-x-plus-a-neg",
            "(min ?x (+ ?x ?a))",
            "(+ ?x ?a)",
            sto_is_const_neg("?a"),
        ),
        sto_rw_cond(
            "min-mul-in-pos",
            "(* (min ?x ?y) ?z)",
            "(min (* ?x ?z) (* ?y ?z))",
            sto_is_const_pos("?z"),
        ),
        sto_rw_cond(
            "min-mul-out-pos",
            "(min (* ?x ?z) (* ?y ?z))",
            "(* (min ?x ?y) ?z)",
            sto_is_const_pos("?z"),
        ),
        sto_rw_cond(
            "min-mul-in-neg",
            "(* (min ?x ?y) ?z)",
            "(max (* ?x ?z) (* ?y ?z))",
            sto_is_const_neg("?z"),
        ),
        sto_rw_cond(
            "min-mul-out-neg",
            "(max (* ?x ?z) (* ?y ?z))",
            "(* (min ?x ?y) ?z)",
            sto_is_const_neg("?z"),
        ),
        sto_rw_cond(
            "min-div-in-pos",
            "(/ (min ?x ?y) ?z)",
            "(min (/ ?x ?z) (/ ?y ?z))",
            sto_is_const_pos("?z"),
        ),
        sto_rw_cond(
            "min-div-out-pos",
            "(min (/ ?x ?z) (/ ?y ?z))",
            "(/ (min ?x ?y) ?z)",
            sto_is_const_pos("?z"),
        ),
        sto_rw_cond(
            "min-div-in-neg",
            "(/ (max ?x ?y) ?z)",
            "(min (/ ?x ?z) (/ ?y ?z))",
            sto_is_const_neg("?z"),
        ),
        sto_rw_cond(
            "min-div-out-neg",
            "(min (/ ?x ?z) (/ ?y ?z))",
            "(/ (max ?x ?y) ?z)",
            sto_is_const_neg("?z"),
        ),
        sto_rw_cond(
            "min-max-const",
            "( min ( max ?x ?c0 ) ?c1 )",
            "?c1",
            sto_compare_c0_c1("?c1", "?c0", "<="),
        ),
        sto_rw_cond(
            "min-div-mul",
            "( min ( * ( / ?x ?c0 ) ?c0 ) ?x )",
            "( * ( / ?x ?c0 ) ?c0 )",
            sto_is_const_pos("?c0"),
        ),
        sto_rw_cond(
            "min-mod-const-to-mod",
            "(min (% ?x ?c0) ?c1)",
            "(% ?x ?c0)",
            sto_compare_c0_c1("?c1", "?c0", ">=a-1"),
        ),
        sto_rw_cond(
            "min-mod-const-to-const",
            "(min (% ?x ?c0) ?c1)",
            "?c1",
            sto_compare_c0_c1("?c1", "?c0", "<=-a+1"),
        ),
        sto_rw_cond(
            "min-max-switch",
            "( min ( max ?x ?c0 ) ?c1 )",
            "( max ( min ?x ?c1 ) ?c0 )",
            sto_compare_c0_c1("?c0", "?c1", "<="),
        ),
        sto_rw_cond(
            "max-min-switch",
            "( max ( min ?x ?c1 ) ?c0 )",
            "( min ( max ?x ?c0 ) ?c1 )",
            sto_compare_c0_c1("?c0", "?c1", "<="),
        ),
        sto_rw("min-consts-or", "( < ( min ?y ?c0 ) ?c1 )", "( || ( < ?y ?c1 ) ( < ?c0 ?c1 ) )"),
        sto_rw("max-consts-and", "( < ( max ?y ?c0 ) ?c1 )", "( && ( < ?y ?c1 ) ( < ?c0 ?c1 ) )"),
        sto_rw("max-consts-or", "( < ?c1 ( max ?y ?c0 ) )", "( || ( < ?c1 ?y ) ( < ?c1 ?c0 ) )"),
        sto_rw_cond(
            "min-consts-div-pos",
            "( min ( * ?x ?a ) ?b )",
            "( * ( min ?x ( / ?b ?a ) ) ?a )",
            sto_compare_c0_c1("?b", "?a", "%0<"),
        ),
        sto_rw_cond(
            "min-min-div-pos",
            "( min ( * ?x ?a ) ( * ?y ?b ) )",
            "( * ( min ?x ( * ?y ( / ?b ?a ) ) ) ?a )",
            sto_compare_c0_c1("?b", "?a", "%0<"),
        ),
        sto_rw_cond(
            "min-consts-div-neg",
            "( min ( * ?x ?a ) ?b )",
            "( * ( max ?x ( / ?b ?a ) ) ?a )",
            sto_compare_c0_c1("?b", "?a", "%0>"),
        ),
        sto_rw_cond(
            "min-min-div-neg",
            "( min ( * ?x ?a ) ( * ?y ?b ) )",
            "( * ( max ?x ( * ?y ( / ?b ?a ) ) ) ?a )",
            sto_compare_c0_c1("?b", "?a", "%0>"),
        ),
    ]
}

pub fn sto_lt() -> Vec<StoRw> {
    vec![
        sto_rw("gt-to-lt", "(> ?x ?z)", "(< ?z ?x)"),
        sto_rw("lt-swap", "(< ?x ?y)", "(< (* -1 ?y) (* -1 ?x))"),
        sto_rw("lt-to-zero", "(< ?a ?a)", "0"),
        sto_rw("lt-swap-in", "(< (+ ?x ?y) ?z)", "(< ?x (- ?z ?y))"),
        sto_rw("lt-swap-out", "(< ?z (+ ?x ?y))", "(< (- ?z ?y) ?x)"),
        sto_rw_cond(
            "lt-x-x-sub-a",
            "(< (- ?a ?y) ?a )",
            "1",
            sto_is_const_pos("?y"),
        ),
        sto_rw_cond("lt-const-pos", "(< 0 ?y )", "1", sto_is_const_pos("?y")),
        sto_rw_cond("lt-const-neg", "(< ?y 0 )", "1", sto_is_const_neg("?y")),
        sto_rw("min-lt-cancel", "( < ( min ?x ?y ) ?x )", "( < ?y ?x )"),
        sto_rw(
            "lt-min-mutual-term",
            "( < ( min ?z ?y ) ( min ?x ?y ) )",
            "( < ?z ( min ?x ?y ) )",
        ),
        sto_rw(
            "lt-max-mutual-term",
            "( < ( max ?z ?y ) ( max ?x ?y ) )",
            "( < ( max ?z ?y ) ?x )",
        ),
        sto_rw_cond(
            "lt-min-term-term+pos",
            "( < ( min ?z ?y ) ( min ?x ( + ?y ?c0 ) ) )",
            "( < ( min ?z ?y ) ?x )",
            sto_is_const_pos("?c0"),
        ),
        sto_rw_cond(
            "lt-max-term-term+pos",
            "( < ( max ?z ( + ?y ?c0 ) ) ( max ?x ?y ) )",
            "( < ( max ?z ( + ?y ?c0 ) ) ?x )",
            sto_is_const_pos("?c0"),
        ),
        sto_rw_cond(
            "lt-min-term+neg-term",
            "( < ( min ?z ( + ?y ?c0 ) ) ( min ?x ?y ) )",
            "( < ( min ?z ( + ?y ?c0 ) ) ?x )",
            sto_is_const_neg("?c0"),
        ),
        sto_rw_cond(
            "lt-max-term+neg-term",
            "( < ( max ?z ?y ) ( max ?x ( + ?y ?c0 ) ) )",
            "( < ( max ?z ?y ) ?x )",
            sto_is_const_neg("?c0"),
        ),
        sto_rw_cond(
            "lt-min-term+cpos",
            "( < ( min ?x ?y ) (+ ?x ?c0) )",
            "1",
            sto_is_const_pos("?c0"),
        ),
        sto_rw("lt-min-max-cancel", "(< (max ?a ?c) (min ?a ?b))", "0"),
        sto_rw_cond(
            "lt-mul-pos-cancel",
            "(< (* ?x ?y) ?z)",
            "(< ?x ( / (- ( + ?z ?y ) 1 ) ?y ) )",
            sto_is_const_pos("?y"),
        ),
        sto_rw_cond(
            "lt-mul-div-cancel",
            "(< ?y (/ ?x ?z))",
            "( < ( - ( * ( + ?y 1 ) ?z ) 1 ) ?x )",
            sto_is_const_pos("?z"),
        ),
        sto_rw_cond(
            "lt-const-mod",
            "(< ?a (% ?x ?b))",
            "1",
            sto_compare_c0_c1("?a", "?b", "<=-a"),
        ),
        sto_rw_cond(
            "lt-const-mod-false",
            "(< ?a (% ?x ?b))",
            "0",
            sto_compare_c0_c1("?a", "?b", ">=a"),
        ),
    ]
}

pub fn sto_ineq() -> Vec<StoRw> {
    vec![sto_rw("ineq-to-eq", "(!= ?x ?y)", "(! (== ?x ?y))")]
}

pub fn sto_modulo() -> Vec<StoRw> {
    vec![
        sto_rw("mod-zero", "(% 0 ?x)", "0"),
        sto_rw("mod-x-x", "(% ?x ?x)", "0"),
        sto_rw("mod-one", "(% ?x 1)", "0"),
        sto_rw_cond(
            "mod-const-add",
            "(% ?x ?c1)",
            "(% (+ ?x ?c1) ?c1)",
            sto_compare_c0_c1("?c1", "?x", "<=a"),
        ),
        sto_rw_cond(
            "mod-const-sub",
            "(% ?x ?c1)",
            "(% (- ?x ?c1) ?c1)",
            sto_compare_c0_c1("?c1", "?x", "<=a"),
        ),
        sto_rw("mod-minus-out", "(% (* ?x -1) ?c)", "(* -1 (% ?x ?c))"),
        sto_rw("mod-minus-in", "(* -1 (% ?x ?c))", "(% (* ?x -1) ?c)"),
        sto_rw("mod-two", "(% (- ?x ?y) 2)", "(% (+ ?x ?y) 2)"),
        sto_rw_cond(
            "mod-consts",
            "( % ( + ( * ?x ?c0 ) ?y ) ?c1 )",
            "( % ?y ?c1 )",
            sto_compare_c0_c1("?c0", "?c1", "%0"),
        ),
        sto_rw_cond(
            "mod-multiple",
            "(% (* ?c0 ?x) ?c1)",
            "0",
            sto_compare_c0_c1("?c0", "?c1", "%0"),
        ),
    ]
}

pub fn sto_andor() -> Vec<StoRw> {
    vec![
        sto_rw(
            "and-over-or",
            "(&& ?a (|| ?b ?c))",
            "(|| (&& ?a ?b) (&& ?a ?c))",
        ),
        sto_rw(
            "or-over-and",
            "(|| ?a (&& ?b ?c))",
            "(&& (|| ?a ?b) (|| ?a ?c))",
        ),
        sto_rw("or-x-and-x-y", "(|| ?x (&& ?x ?y))", "?x"),
    ]
}

/// All stochastic rules combined.
pub fn all_sto_rules() -> Vec<StoRw> {
    let rule_sets: Vec<Vec<StoRw>> = vec![
        sto_add(),
        sto_mul(),
        sto_sub(),
        sto_div(),
        sto_eq(),
        sto_and(),
        sto_or(),
        sto_not(),
        sto_max(),
        sto_min(),
        sto_lt(),
        sto_ineq(),
        sto_modulo(),
        sto_andor(),
    ];
    rule_sets.into_iter().flatten().collect()
}
