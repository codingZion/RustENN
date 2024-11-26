use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn Error>> {
    // Input CSV file path
    let input_path = "params.csv";

    // Open the CSV file
    let input_file = File::open(input_path)?;
    let mut reader = ReaderBuilder::new().from_reader(BufReader::new(input_file));

    // Read the header and the first row
    let headers = reader.headers()?.clone();
    let mut rows = reader.records();

    if let Some(Ok(row)) = rows.next() {
        // Combine the headers and row values into a LaTeX-compatible format
        let latex_output: String = headers
            .iter()
            .zip(row.iter())
            .map(|(header, value)| format!("{}: \\texttt{{{}}}", escape_latex(header), escape_latex(value)))
            .collect::<Vec<String>>()
            .join(", ");

        // Print the LaTeX-compatible string
        println!("{}", latex_output);
    } else {
        println!("The CSV file is empty or formatted incorrectly.");
    }

    Ok(())
}

// Escape special LaTeX characters
fn escape_latex(input: &str) -> String {
    input
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("^", "\\textasciicircum{}")
        .replace("~", "\\textasciitilde{}")
}
