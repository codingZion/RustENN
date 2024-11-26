use csv::{ReaderBuilder, WriterBuilder};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn main() -> Result<(), Box<dyn Error>> {
    // Input and output file paths
    let input_path = "stats_single.csv";
    let output_path = "stats.csv";

    // Open the input and output files
    let input_file = File::open(input_path)?;
    let output_file = File::create(output_path)?;

    // Set up CSV reader and writer
    let mut reader = ReaderBuilder::new().from_reader(BufReader::new(input_file));
    let mut writer = WriterBuilder::new().from_writer(BufWriter::new(output_file));

    // Write the new header with "generation" as the first column
    let mut headers = reader.headers()?.clone();
    let mut new_headers = vec!["generation".to_string()];
    new_headers.extend(headers.iter().map(String::from));
    writer.write_record(&new_headers)?;

    // Process rows and add the index as the "generation" value
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        let mut new_row = vec![i.to_string()];
        new_row.extend(record.iter().map(String::from));
        writer.write_record(&new_row)?;
    }

    println!("CSV file updated successfully. Output written to {}", output_path);
    Ok(())
}
