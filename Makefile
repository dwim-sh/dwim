# Builds dwim and installs it and its man page under PREFIX:
#
#   make install                     # in ~/.local
#   make install PREFIX=/usr/local   # system-wide, which may need sudo
#
# `make` builds as you, so `make && sudo make install PREFIX=/usr/local`
# builds nothing as root.
#
# `make spec` checks the Lean specs in specs/, their definitions, their
# examples and their proofs, and makes the pages that show them: the
# specs, and an animation of a session played through the spec of
# compaction. It prints where they are. `make check` does that too, and
# checks that the code does what the specs say, with difftest.
#
# Each step prints one line such as `  INSTALL ~/.local/bin/dwim`.
# `make V=1` prints the commands instead.
#
# To see a list of the targets execute "make help".

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
MANDIR ?= $(PREFIX)/share/man

BINARY = target/release/dwim
SOURCES = Cargo.toml Cargo.lock rust-toolchain.toml build.rs lib.rs \
	$(wildcard */*.rs */*/*.rs */*/*/*.rs gpu/*/kernels/*)

# $(Q) hides a command, and $(call msg,VERB,what) announces it. With V=1
# the command is shown instead; with `make -s` nothing is.
ifeq ($(V),1)
  Q =
  msg =
else
  Q = @
  msg = @printf '  %-8s %s\n' '$(1)' '$(2)'
endif
# $(call quiet,command) shows what the command printed only if it fails.
quiet = out=$$($(1) 2>&1) || { printf '%s\n' "$$out"; exit 1; }

ifneq ($(findstring s,$(firstword -$(MAKEFLAGS))),)
  msg =
endif

all: $(BINARY)

# Brief documentation of the targets, shown by `make help`.
help:
	@echo '  all       - Build the binary (release)'
	@echo '  install   - Install the binary and the man page under PREFIX'
	@echo '  uninstall - Remove what `make install` installed'
	@echo '  test      - Run the test suite'
	@echo '  spec      - Check the Lean specs, and make their pages'
	@echo '  check     - Check the Lean specs, and the code against them'
	@echo '  clean     - Remove build artifacts'

$(BINARY): $(SOURCES)
	$(call msg,CARGO,$@)
	$(Q)cargo build --release --locked
	$(Q)touch $@

install: all
	$(call msg,INSTALL,$(DESTDIR)$(BINDIR)/dwim)
	$(Q)install -d $(DESTDIR)$(BINDIR)
	$(Q)install -m 755 $(BINARY) $(DESTDIR)$(BINDIR)/dwim
	$(call msg,INSTALL,$(DESTDIR)$(MANDIR)/man1/dwim.1)
	$(Q)install -d $(DESTDIR)$(MANDIR)/man1
	$(Q)install -m 644 man/dwim.1 $(DESTDIR)$(MANDIR)/man1/dwim.1

uninstall:
	$(call msg,RM,$(DESTDIR)$(BINDIR)/dwim)
	$(Q)rm -f $(DESTDIR)$(BINDIR)/dwim
	$(call msg,RM,$(DESTDIR)$(MANDIR)/man1/dwim.1)
	$(Q)rm -f $(DESTDIR)$(MANDIR)/man1/dwim.1

test:
	$(call msg,TEST,dwim)
	$(Q)cargo test

check: difftest-compaction difftest-harness spec

# `make spec` checks the Lean specs and their proofs, and makes their
# pages, without running the difftests. It ends by saying what to open.
PAGES = specs/.lake/build/literate-html
spec:
	$(call msg,LAKE,$(PAGES)/index.html)
	@# Verso does not notice changes to images and style.css: without its
	@# trace, it makes the pages again, which takes a second.
	$(Q)rm -f $(PAGES).trace
	$(Q)cd specs && lake --quiet query :literateHtml > /dev/null
	$(call msg,LAKE,$(PAGES)/compaction.html)
	$(Q)cd specs && lake --quiet exe compaction-trace > .lake/build/literate-html/compaction-trace.js
	$(Q)cp specs/compaction.html $(PAGES)/compaction.html
	@printf '\nOpen in a browser:\n'
	@printf '  The specs:      file://%s/index.html\n' "$(abspath $(PAGES))"
	@printf '  The animation:  file://%s/compaction.html\n' "$(abspath $(PAGES))"

# `make difftest-SUBJECT` checks one subject against its reference, as
# `difftest SUBJECT` does: `make difftest-compaction`, `make difftest-model`.
difftest-%:
	$(call msg,DIFFTEST,$*)
	$(Q)$(call quiet,cargo run --quiet --bin difftest $*)

clean:
	$(call msg,CLEAN,target)
	$(Q)cargo clean

.PHONY: help all install uninstall test check spec clean
