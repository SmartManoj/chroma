use std::fmt::Debug;

use crate::errors::{ChromaError, ErrorCodes};

use super::{Component, ComponentSender, Handler};
use async_trait::async_trait;
use thiserror::Error;

// Receiver Traits
#[async_trait]
pub(crate) trait ReceiverForMessage<M>:
    Send + Sync + Debug + ReceiverForMessageClone<M>
{
    async fn send(
        &self,
        message: M,
        tracing_context: Option<tracing::Span>,
    ) -> Result<(), ChannelError>;
}

pub(crate) trait ReceiverForMessageClone<M> {
    fn clone_box(&self) -> Box<dyn ReceiverForMessage<M>>;
}

impl<M> Clone for Box<dyn ReceiverForMessage<M>> {
    fn clone(&self) -> Box<dyn ReceiverForMessage<M>> {
        self.clone_box()
    }
}

impl<T, M> ReceiverForMessageClone<M> for T
where
    T: 'static + ReceiverForMessage<M> + Clone,
{
    fn clone_box(&self) -> Box<dyn ReceiverForMessage<M>> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl<C, M> ReceiverForMessage<M> for ComponentSender<C>
where
    C: Component + Handler<M>,
    M: Send + Debug + 'static,
{
    async fn send(
        &self,
        message: M,
        tracing_context: Option<tracing::Span>,
    ) -> Result<(), ChannelError> {
        self.wrap_and_send(message, tracing_context).await
    }
}

// Errors
#[derive(Error, Debug)]
pub enum ChannelError {
    #[error("Failed to send message")]
    SendError,
}

impl ChromaError for ChannelError {
    fn code(&self) -> ErrorCodes {
        ErrorCodes::Internal
    }
}
