#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Brand {
    Amd,
    Nvidia,
}

impl Brand {
    pub fn platform_name(&self) -> &'static str {
        match self {
            Brand::Nvidia => "Intel(R) CPU Runtime for OpenCL(TM) Applications",
            Brand::Amd => "AMD Accelerated Parallel Processing",
        }
    }

    pub fn get_bus_id(&self, d: ocl::Device) -> ocl::Result<u32> {
        match self {
            Brand::Nvidia => {
                const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
                let result = d.info_raw(CL_DEVICE_PCI_BUS_ID_NV)?;
                Ok(u32::from_le_bytes(result[..].try_into().unwrap()))
            }
            Brand::Amd => panic!("Not supported!"),
        }
    }
}

pub fn find_platform(platform_name: &str) -> ocl::Result<Option<ocl::Platform>> {
    Ok(ocl::Platform::list().into_iter().find(|&p| match p.name() {
        Ok(p) => p == platform_name.to_string(),
        Err(_) => false,
    }))
}

pub struct Buffer<T> {
    buffer: ocl::Buffer<u8>,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    bus_id: u32,
    platform: ocl::Platform,
    device: ocl::Device,
}

impl Device {
    pub fn bus_id(&self) -> u32 {
        self.bus_id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn brand(&self) -> Brand {
        self.brand
    }
    pub fn by_brand(brand: Brand) -> ocl::Result<Vec<Device>> {
        match find_platform(brand.platform_name())? {
            Some(plat) => ocl::Device::list_all(plat)?
                .into_iter()
                .map(|d| {
                    (|| -> ocl::Result<Device> {
                        Ok(Device {
                            brand,
                            name: d.name()?,
                            bus_id: 0, //brand.get_bus_id(d)?,
                            platform: plat,
                            device: d,
                        })
                    })()
                })
                .collect(),
            None => Ok(Vec::new()),
        }
    }
}

pub struct Program {
    device: Device,
    program: ocl::Program,
    queue: ocl::Queue,
}

#[derive(thiserror::Error, Debug)]
pub enum ProgramError {
    #[error("OclCore Error: {0}")]
    OclCoreError(#[from] ocl::OclCoreError),
    #[error("Ocl Error: {0}")]
    OclError(#[from] ocl::Error),
    #[error("Program info not available!")]
    ProgramInfoNotAvailable(ocl::enums::ProgramInfo),
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),
}

impl Program {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn from_opencl(device: &Device, src: &str) -> Result<Program, ProgramError> {
        //if std::path::Path::exists(&cached) {
        //    let bin = std::fs::read(cached)?;
        //    Program::from_binary(device, bin)
        //} else {
        println!("{:?} {:?}", device.platform, device.device);
        let context = ocl::Context::builder().platform(device.platform).build()?;
        let program = ocl::Program::builder()
            .src(src)
            .devices(ocl::builders::DeviceSpecifier::Single(device.device))
            .build(&context)?;
        let queue = ocl::Queue::new(&context, device.device, None)?;
        let prog = Program {
            program,
            queue,
            device: device.clone(),
        };
        //std::fs::write(cached, prog.to_binary()?)?;
        Ok(prog)
        //}
    }
    pub fn to_binary(&self) -> Result<Vec<u8>, ProgramError> {
        match self.program.info(ocl::enums::ProgramInfo::Binaries)? {
            ocl::enums::ProgramInfoResult::Binaries(bins) => Ok(bins[0].clone()),
            _ => Err(ProgramError::ProgramInfoNotAvailable(
                ocl::enums::ProgramInfo::Binaries,
            )),
        }
    }
    pub fn from_binary(device: &Device, bin: Vec<u8>) -> Result<Program, ProgramError> {
        let context = ocl::Context::builder()
            .platform(device.platform)
            .devices(device.device)
            .build()?;
        let bins = vec![&bin[..]];
        let program = ocl::Program::builder()
            .binaries(&bins)
            .devices(ocl::builders::DeviceSpecifier::Single(device.device))
            .build(&context)?;
        let queue = ocl::Queue::new(&context, device.device, None)?;
        Ok(Program {
            device: device.clone(),
            program,
            queue,
        })
    }
    pub fn create_buffer<T>(&self, length: usize) -> Result<Buffer<T>, ProgramError> {
        assert!(length > 0);
        let buff = ocl::Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(length * std::mem::size_of::<T>())
            .build()?;
        buff.write(&vec![0u8]).enq()?;
        Ok(Buffer::<T> {
            buffer: buff,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn create_buffer_from_slice<T>(&self, vals: &[T]) -> Result<Buffer<T>, ProgramError> {
        let mut buf = self.create_buffer::<T>(vals.len())?;
        buf.write_from(vals)?;
        Ok(buf)
    }

    pub fn create_kernel(&self, name: &str, gws: usize, lws: usize) -> Kernel<'_> {
        let mut builder = ocl::Kernel::builder();
        builder.name(name);
        builder.program(&self.program);
        builder.queue(self.queue.clone());
        builder.global_work_size([gws]);
        builder.local_work_size([lws]);
        Kernel::<'_> { builder }
    }
}

pub trait KernelArgument<'a> {
    fn push(&self, kernel: &mut Kernel<'a>);
}

impl<'a, T> KernelArgument<'a> for &'a Buffer<T> {
    fn push(&self, kernel: &mut Kernel<'a>) {
        kernel.builder.arg(&self.buffer);
    }
}

impl<T: ocl::OclPrm> KernelArgument<'_> for T {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.arg(self.clone());
    }
}

pub struct LocalBuffer<T> {
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T> LocalBuffer<T> {
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> KernelArgument<'_> for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .builder
            .arg_local::<u8>(self.length * std::mem::size_of::<T>());
    }
}

#[derive(Debug)]
pub struct Kernel<'a> {
    builder: ocl::builders::KernelBuilder<'a>,
}

impl<'a> Kernel<'a> {
    pub fn arg<T: KernelArgument<'a>>(mut self, t: T) -> Self {
        t.push(&mut self);
        self
    }
    pub fn run(self) -> Result<(), ProgramError> {
        let kern = self.builder.build()?;
        unsafe {
            kern.enq()?;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! call_kernel {
    ($kernel:expr, $($arg:expr),*) => {{
        $kernel
        $(.arg($arg))*
        .run()
    }};
}

impl<T> Buffer<T> {
    pub fn length(&self) -> usize {
        self.buffer.len() / std::mem::size_of::<T>()
    }

    pub fn write_from(&mut self, data: &[T]) -> Result<(), ProgramError> {
        assert!(data.len() <= self.length());
        self.buffer
            .write(unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const T as *const u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            })
            .enq()?;
        Ok(())
    }

    pub fn read_into(&self, data: &mut [T]) -> Result<(), ProgramError> {
        assert!(data.len() <= self.length());
        self.buffer
            .read(unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr() as *mut T as *mut u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            })
            .enq()?;
        Ok(())
    }
}
