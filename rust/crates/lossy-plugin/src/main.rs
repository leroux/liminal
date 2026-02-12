#[cfg(target_os = "macos")]
fn set_macos_icon() {
    use cocoa::base::{id, nil};
    use cocoa::foundation::NSData;
    use objc::{class, msg_send, sel, sel_impl};

    static ICON_PNG: &[u8] = include_bytes!("../../../../icons/lossy.png");
    unsafe {
        let app: id = msg_send![class!(NSApplication), sharedApplication];
        let data = NSData::dataWithBytes_length_(
            nil,
            ICON_PNG.as_ptr() as *const std::os::raw::c_void,
            ICON_PNG.len() as u64,
        );
        let image: id = msg_send![class!(NSImage), alloc];
        let image: id = msg_send![image, initWithData: data];
        let _: () = msg_send![app, setApplicationIconImage: image];
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    set_macos_icon();

    nih_plug::nih_export_standalone::<lossy_plugin::LossyPlugin>();
}
