//! Secrets kept from one run to the next, in the user's login keychain.
//!
//! The keychain keeps a secret encrypted under the user's password, and
//! gives it only to the program that put it there, as its code signature
//! names it: anything else the user runs that asks for it, `security`
//! included, has the user asked first. A program that is not signed by a
//! developer is named by the hash of its code, so one built anew is
//! another program to the keychain, and has the user asked once.
//!
//! This is the login keychain, a file, and not the data protection
//! keychain, which takes entitlements a command line program has none of.
//!
//! Where the keychain cannot be used, as in a session over SSH, whose
//! keychain is locked, a secret is kept in a file of the user's own, as
//! on Linux. A secret found in such a file is moved to the keychain once
//! it can be.

use std::{
    ffi::{c_char, c_void},
    io, ptr, slice,
};

use crate::sys::unix::secret_file;

pub use crate::sys::unix::secret_file::Place;

/// Keeps `secret` as `account` of `service`, and returns where: in the
/// keychain, or in a file where the keychain cannot be used.
pub fn store(service: &str, account: &str, secret: &str) -> io::Result<Place> {
    match keychain_store(service, account, secret) {
        Ok(()) => {
            // The keychain's is the one read from here on.
            let _ = secret_file::delete(service, account);
            Ok(Place::Keychain)
        }
        Err(_) => secret_file::store(service, account, secret),
    }
}

/// The secret kept as `account` of `service`, if there is one. The user
/// is asked whether this program may have it, if another put it there.
pub fn load(service: &str, account: &str) -> io::Result<Option<String>> {
    match keychain_load(service, account) {
        Ok(Some(secret)) => return Ok(Some(secret)),
        Ok(None) => {}
        // A keychain that cannot be used leaves the file; one that
        // refused its secret is an answer.
        Err(_) if secret_file::exists(service, account) => {}
        Err(status) => return Err(error(status)),
    }
    let secret = secret_file::load(service, account)?;
    if let Some(secret) = &secret
        && keychain_store(service, account, secret).is_ok()
    {
        let _ = secret_file::delete(service, account);
    }
    Ok(secret)
}

/// Whether a secret is kept as `account` of `service`, which the user is
/// never asked about.
pub fn exists(service: &str, account: &str) -> bool {
    let query = Dictionary::item(service, account).with(
        // SAFETY: the constants are Security's, and live as long as the
        // program does.
        unsafe { kSecReturnAttributes },
        unsafe { kCFBooleanTrue },
    );
    let mut found = ptr::null();
    // SAFETY: the query is a dictionary of Security's own keys, and
    // `found` is written with an object this owns, or left null.
    let status = unsafe { SecItemCopyMatching(query.get().0, &mut found) };
    drop(Owned(found));
    status == SUCCESS || secret_file::exists(service, account)
}

/// Forgets the secret kept as `account` of `service`, if there is one.
pub fn delete(service: &str, account: &str) -> io::Result<()> {
    let deleted = keychain_delete(service, account);
    secret_file::delete(service, account)?;
    deleted.map_err(error)
}

/// Keeps the user's other processes from reading this one's memory: a
/// debugger can no longer attach to it. A program this one starts is not
/// kept so.
pub fn guard_memory() -> io::Result<()> {
    // SAFETY: PT_DENY_ATTACH takes no address and no data.
    match unsafe { libc::ptrace(libc::PT_DENY_ATTACH, 0, ptr::null_mut::<c_char>(), 0) } {
        0 => Ok(()),
        _ => Err(io::Error::last_os_error()),
    }
}

/// Puts `secret` in the keychain as `account` of `service`, in place of
/// the one there, if any: the new item is this program's, whichever
/// program the old one was.
fn keychain_store(service: &str, account: &str, secret: &str) -> Result<(), Status> {
    keychain_delete(service, account)?;
    let label = Owned::string(&format!("{service}: {account}"));
    let data = Owned::data(secret.as_bytes());
    // SAFETY: the constants are Security's, and live as long as the
    // program does.
    let item = Dictionary::item(service, account)
        .with(unsafe { kSecAttrLabel }, label.0)
        .with(unsafe { kSecValueData }, data.0);
    // SAFETY: the attributes are a dictionary of Security's own keys, and
    // no result is asked for.
    match unsafe { SecItemAdd(item.get().0, ptr::null_mut()) } {
        SUCCESS => Ok(()),
        status => Err(status),
    }
}

/// The secret in the keychain as `account` of `service`, if there is one.
fn keychain_load(service: &str, account: &str) -> Result<Option<String>, Status> {
    // SAFETY: the constants are Security's, and live as long as the
    // program does.
    let query = Dictionary::item(service, account)
        .with(unsafe { kSecReturnData }, unsafe { kCFBooleanTrue })
        .with(unsafe { kSecMatchLimit }, unsafe { kSecMatchLimitOne });
    let mut found = ptr::null();
    // SAFETY: the query is a dictionary of Security's own keys, and
    // `found` is written with an object this owns, or left null.
    let status = unsafe { SecItemCopyMatching(query.get().0, &mut found) };
    let found = Owned(found);
    match status {
        SUCCESS if !found.0.is_null() => {
            // SAFETY: asked for one item's data, Security returns a
            // CFData, whose bytes live as long as it does.
            let bytes = unsafe {
                let len = CFDataGetLength(found.0);
                match len {
                    0 => &[][..],
                    _ => slice::from_raw_parts(CFDataGetBytePtr(found.0), len as usize),
                }
            };
            let secret = String::from_utf8_lossy(bytes).trim().to_string();
            Ok((!secret.is_empty()).then_some(secret))
        }
        SUCCESS | NOT_FOUND => Ok(None),
        status => Err(status),
    }
}

/// Takes `account` of `service` out of the keychain, if it is there.
fn keychain_delete(service: &str, account: &str) -> Result<(), Status> {
    let query = Dictionary::item(service, account);
    // SAFETY: the query is a dictionary of Security's own keys.
    match unsafe { SecItemDelete(query.get().0) } {
        SUCCESS | NOT_FOUND => Ok(()),
        status => Err(status),
    }
}

/// What the keychain answered `status` for, as an error to show.
fn error(status: Status) -> io::Error {
    let why = match status {
        USER_CANCELED | AUTH_FAILED => "the keychain was not allowed to give it".to_string(),
        INTERACTION_NOT_ALLOWED => "the keychain is locked".to_string(),
        status => format!("the keychain answered {status}"),
    };
    io::Error::other(why)
}

/// What Security answers a call with.
type Status = i32;

const SUCCESS: Status = 0;
/// The user dismissed the question of whether the program may have it.
const USER_CANCELED: Status = -128;
/// The user answered that it may not.
const AUTH_FAILED: Status = -25293;
const NOT_FOUND: Status = -25300;
/// There is no keychain to ask.
#[cfg(test)]
const NO_KEYCHAIN: Status = -25307;
/// The user would have to be asked, and cannot be: the keychain is
/// locked, in a session with no screen.
const INTERACTION_NOT_ALLOWED: Status = -25308;

/// A Core Foundation object this owns, released when dropped; or none.
struct Owned(Object);

impl Owned {
    /// `text` as a CFString.
    fn string(text: &str) -> Self {
        // SAFETY: the bytes are `text`'s, UTF-8, and copied by the call.
        Self(unsafe {
            CFStringCreateWithBytes(ptr::null(), text.as_ptr(), text.len() as isize, UTF8, 0)
        })
    }

    /// `bytes` as a CFData.
    fn data(bytes: &[u8]) -> Self {
        // SAFETY: the bytes are copied by the call.
        Self(unsafe { CFDataCreate(ptr::null(), bytes.as_ptr(), bytes.len() as isize) })
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the object is one this owns, released once.
            unsafe { CFRelease(self.0) };
        }
    }
}

/// The keys and values of a query or an item, for a CFDictionary made of
/// them when it is asked with.
struct Dictionary {
    keys: Vec<Object>,
    values: Vec<Object>,
    /// The strings made for it, which live as long as it does.
    _strings: Vec<Owned>,
}

impl Dictionary {
    /// What names the generic password `account` of `service`.
    fn item(service: &str, account: &str) -> Self {
        let strings = vec![Owned::string(service), Owned::string(account)];
        // SAFETY: the constants are Security's, and live as long as the
        // program does.
        let (keys, values) = unsafe {
            (
                vec![kSecClass, kSecAttrService, kSecAttrAccount],
                vec![kSecClassGenericPassword, strings[0].0, strings[1].0],
            )
        };
        Self {
            keys,
            values,
            _strings: strings,
        }
    }

    /// The same with `value` for `key`, where `value` outlives it.
    fn with(mut self, key: Object, value: Object) -> Self {
        self.keys.push(key);
        self.values.push(value);
        self
    }

    /// The CFDictionary of it, which keeps its keys and values itself.
    fn get(&self) -> Owned {
        // SAFETY: the keys and values are as many Core Foundation
        // objects, alive for the call, which the dictionary retains
        // through the callbacks for such objects.
        Owned(unsafe {
            CFDictionaryCreate(
                ptr::null(),
                self.keys.as_ptr(),
                self.values.as_ptr(),
                self.keys.len() as isize,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            )
        })
    }
}

/// A Core Foundation object, as a call takes it.
type Object = *const c_void;

/// A structure of Core Foundation's that is only ever pointed to.
#[repr(C)]
struct Opaque {
    _private: [u8; 0],
}

/// The encoding of a CFString made of UTF-8.
const UTF8: u32 = 0x0800_0100;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeDictionaryKeyCallBacks: Opaque;
    static kCFTypeDictionaryValueCallBacks: Opaque;
    static kCFBooleanTrue: Object;

    fn CFDictionaryCreate(
        allocator: Object,
        keys: *const Object,
        values: *const Object,
        count: isize,
        key_callbacks: *const Opaque,
        value_callbacks: *const Opaque,
    ) -> Object;
    fn CFStringCreateWithBytes(
        allocator: Object,
        bytes: *const u8,
        len: isize,
        encoding: u32,
        external: u8,
    ) -> Object;
    fn CFDataCreate(allocator: Object, bytes: *const u8, len: isize) -> Object;
    fn CFDataGetBytePtr(data: Object) -> *const u8;
    fn CFDataGetLength(data: Object) -> isize;
    fn CFRelease(object: Object);
}

#[link(name = "Security", kind = "framework")]
unsafe extern "C" {
    static kSecClass: Object;
    static kSecClassGenericPassword: Object;
    static kSecAttrService: Object;
    static kSecAttrAccount: Object;
    static kSecAttrLabel: Object;
    static kSecValueData: Object;
    static kSecReturnData: Object;
    static kSecReturnAttributes: Object;
    static kSecMatchLimit: Object;
    static kSecMatchLimitOne: Object;

    fn SecItemAdd(attributes: Object, result: *mut Object) -> Status;
    fn SecItemCopyMatching(query: Object, result: *mut Object) -> Status;
    fn SecItemDelete(query: Object) -> Status;
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;

    #[test]
    fn a_secret_put_in_the_keychain_is_read_back_until_it_is_deleted() {
        let (service, account) = ("dwim-test", process::id().to_string());
        match keychain_store(service, &account, "first") {
            // A machine with no keychain to use, as one reached over SSH.
            Err(NO_KEYCHAIN | INTERACTION_NOT_ALLOWED) => return,
            stored => stored.unwrap(),
        }
        keychain_store(service, &account, "second").unwrap();
        assert_eq!(
            keychain_load(service, &account).unwrap().as_deref(),
            Some("second")
        );
        assert!(exists(service, &account));
        keychain_delete(service, &account).unwrap();
        assert_eq!(keychain_load(service, &account).unwrap(), None);
        assert!(!exists(service, &account));
    }
}
