use anyhow::*;
use tch::Tensor;

fn test() -> Result<()> {
    //let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let model = tch::jit::CModule::load("./discard_sl.pt")?;
    println!("meuh");
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t2 = model.forward_ts(&[t]);
    for x in t2.iter() {
        println!("{:?}", x);
    }
    Ok(())
}

fn main() {
    match test() {
        Err(e) => println!("{e:?}"),
        _ => ()
    }
}
