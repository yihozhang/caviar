use crate::structs::{PaperResult, ResultStructure};
use csv::Writer;
use std::error::Error;
use std::fs::OpenOptions;

///Writes the results (a vector of ResultStructure) into a CSV file.
#[allow(dead_code)]
pub fn write_results(path: &str, results: &Vec<ResultStructure>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;

    for result in results {
        wtr.serialize(result)?;
    }

    wtr.flush()?;

    Ok(())
}

///Writes the paper results (a vector of PaperResult) into a CSV file.
#[allow(dead_code)]
pub fn write_results_paper(path: &str, results: &Vec<PaperResult>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;

    for result in results {
        wtr.serialize(result)?;
    }

    wtr.flush()?;

    Ok(())
}

/// Creates a fresh CSV writer at `path` (truncates any existing file).
/// Use this at the start of a run, then call `append_result` for each result.
pub fn create_writer(path: &str) -> Result<Writer<std::fs::File>, Box<dyn Error>> {
    let wtr = Writer::from_path(path)?;
    Ok(wtr)
}

/// Serializes a single `ResultStructure` to an open writer and flushes immediately.
pub fn append_result(
    wtr: &mut Writer<std::fs::File>,
    result: &ResultStructure,
) -> Result<(), Box<dyn Error>> {
    wtr.serialize(result)?;
    wtr.flush()?;
    Ok(())
}

/// Serializes a single `PaperResult` to an open writer and flushes immediately.
#[allow(dead_code)]
pub fn append_result_paper(
    wtr: &mut Writer<std::fs::File>,
    result: &PaperResult,
) -> Result<(), Box<dyn Error>> {
    wtr.serialize(result)?;
    wtr.flush()?;
    Ok(())
}
